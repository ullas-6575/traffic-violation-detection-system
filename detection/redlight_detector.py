"""TVS-8: Bidirectional red-light crossing detector with retroactive check."""

from __future__ import annotations

from datetime import datetime

import supervision as sv

from .config import (
    DIRECTION_FRAMES, DIRECTION_MIN_MOVE, MIN_TRACK_FRAMES, STALE_TRACK_FRAMES,
)
from .models import Direction, SignalState


class RedLightDetector:
    """
    Red light violation detector for bidirectional traffic.
    Uses a Coordinate History Buffer to catch fast vehicles that cross
    the stop line before their direction is classified (retroactive check).

    Signal-per-frame fix: stores the signal state alongside each history
    entry so retroactive checks use the signal AT THE TIME OF CROSSING,
    not the current frame's signal.

    Ported from TVS-8 (red_light_bi.py).
    """

    def __init__(self, stop_line_y: int):
        self.stop_line_y = stop_line_y
        self.tracks:     dict = {}
        self.violations: list = []

    # ── Track management ──────────────────────────────────────────────────────

    def _init_track(self, track_id: int, frame_num: int):
        self.tracks[track_id] = {
            "direction":       Direction.UNKNOWN,
            "history":         [],  # (frame_num, top_y, bottom_y, center_y, signal_state)
            "last_edge_y":     None,
            "frames_tracked":  0,
            "last_seen_frame": frame_num,
            "violated":        False,
            "violation_frame": None,
        }

    def _classify_direction(self, history: list) -> Direction:
        """
        Compare first and last center_y in the history buffer.
        Negative delta → moving UP. Positive delta → moving DOWN.
        """
        if len(history) < DIRECTION_FRAMES:
            return Direction.UNKNOWN

        # History tuples: (frame_num, top_y, bottom_y, center_y, signal_state)
        # Index 3 is center_y
        delta = history[-1][3] - history[0][3]
        if abs(delta) < DIRECTION_MIN_MOVE:
            return Direction.UNKNOWN
        return Direction.UP if delta < 0 else Direction.DOWN

    def _retroactive_check(self, track: dict) -> Optional[dict]:
        """
        Scan history buffer for a crossing that happened during the
        direction classification blind spot.
        Uses the signal state STORED AT THAT FRAME — not the current signal.
        """
        direction = track["direction"]
        for i in range(1, len(track["history"])):
            pf, pt, pb, _, ps = track["history"][i-1]
            cf, ct, cb, _, cs = track["history"][i]
            # Only flag if signal was RED at the time of crossing
            if cs != SignalState.RED:
                continue
            if direction == Direction.UP and pt > self.stop_line_y >= ct:
                return {"frame": cf, "edge_label": "top_y",
                        "edge_val": ct, "signal": cs.value}
            if direction == Direction.DOWN and pb < self.stop_line_y <= cb:
                return {"frame": cf, "edge_label": "bottom_y",
                        "edge_val": cb, "signal": cs.value}
        return None

    # ── Main update ───────────────────────────────────────────────────────────

    def process(self, detections: sv.Detections,
                signal: SignalState, frame_num: int) -> list:
        """
        Process detections for red light violations.
        Returns list of violation event dicts (with bbox for rule engine).
        """
        events     = []
        active_ids = set()

        if detections.tracker_id is None:
            self._cleanup(frame_num, active_ids)
            return events

        active_ids = set(int(tid) for tid in detections.tracker_id)

        for i, track_id in enumerate(detections.tracker_id):
            track_id = int(track_id)
            x1, y1, x2, y2 = detections.xyxy[i]
            top_y    = float(y1)
            bottom_y = float(y2)
            center_y = (top_y + bottom_y) / 2
            bbox     = [float(x1), float(y1), float(x2), float(y2)]

            # Init new track
            if track_id not in self.tracks:
                self._init_track(track_id, frame_num)

            track = self.tracks[track_id]
            track["frames_tracked"]  += 1
            track["last_seen_frame"]  = frame_num

            # 1. ALWAYS ADD TO HISTORY BUFFER (with signal state)
            track["history"].append((frame_num, top_y, bottom_y, center_y, signal))

            # 2. CLASSIFY DIRECTION
            if track["direction"] == Direction.UNKNOWN:
                track["direction"] = self._classify_direction(track["history"])

                # The exact moment direction is found:
                if track["direction"] != Direction.UNKNOWN:
                    print(f"  Track #{track_id} classified as {track['direction'].value}")
                    track["last_edge_y"] = (top_y if track["direction"] == Direction.UP
                                            else bottom_y)

                    # 3. RUN RETROACTIVE CHECK ON THE BUFFER
                    retro = self._retroactive_check(track)
                    if retro and not track["violated"]:
                        track["violated"]        = True
                        track["violation_frame"]  = retro["frame"]

                        v = {
                            "track_id":       track_id,
                            "frame":          retro["frame"],
                            "timestamp":      datetime.now().isoformat(),
                            "signal_state":   retro["signal"],
                            "direction":      track["direction"].value,
                            "stop_line_y":    self.stop_line_y,
                            retro["edge_label"]: round(retro["edge_val"], 1),
                            "frames_tracked": track["frames_tracked"],
                            "bbox":           bbox,
                        }
                        self.violations.append(v)
                        events.append(v)
                    continue  # Skip standard check this frame

            direction = track["direction"]

            # Skip standard checks until direction is known
            if direction == Direction.UNKNOWN:
                continue

            # 4. STANDARD REAL-TIME CHECK
            if signal != SignalState.RED:
                track["last_edge_y"] = (top_y if direction == Direction.UP
                                        else bottom_y)
                continue

            # Cooldown check
            if track["violated"]:
                if frame_num - track["violation_frame"] <= COOLDOWN_FRAMES:
                    track["last_edge_y"] = (top_y if direction == Direction.UP
                                            else bottom_y)
                    continue
                else:
                    track["violated"] = False

            # Minimum track frames
            if track["frames_tracked"] < MIN_TRACK_FRAMES:
                track["last_edge_y"] = (top_y if direction == Direction.UP
                                        else bottom_y)
                continue

            last_edge = track["last_edge_y"]
            if last_edge is None:
                track["last_edge_y"] = (top_y if direction == Direction.UP
                                        else bottom_y)
                continue

            violated = False
            ev_edge_label, ev_edge_val = "", 0.0

            if direction == Direction.UP:
                if top_y <= self.stop_line_y < last_edge:
                    violated       = True
                    ev_edge_label  = "top_y"
                    ev_edge_val    = top_y
                track["last_edge_y"] = top_y

            elif direction == Direction.DOWN:
                if bottom_y >= self.stop_line_y > last_edge:
                    violated       = True
                    ev_edge_label  = "bottom_y"
                    ev_edge_val    = bottom_y
                track["last_edge_y"] = bottom_y

            if violated:
                track["violated"]        = True
                track["violation_frame"]  = frame_num

                v = {
                    "track_id":       track_id,
                    "frame":          frame_num,
                    "timestamp":      datetime.now().isoformat(),
                    "signal_state":   signal.value,
                    "direction":      direction.value,
                    "stop_line_y":    self.stop_line_y,
                    ev_edge_label:    round(ev_edge_val, 1),
                    "frames_tracked": track["frames_tracked"],
                    "bbox":           bbox,
                }
                self.violations.append(v)
                events.append(v)

        self._cleanup(frame_num, active_ids)
        return events

    def _cleanup(self, frame_num: int, active_ids: set):
        stale = [tid for tid, t in self.tracks.items()
                 if tid not in active_ids
                 and frame_num - t["last_seen_frame"] > STALE_TRACK_FRAMES]
        for tid in stale:
            del self.tracks[tid]

    def is_violated(self, track_id: int) -> bool:
        t = self.tracks.get(int(track_id))
        return t is not None and t["violated"]

    def get_direction(self, track_id: int) -> Direction:
        t = self.tracks.get(int(track_id))
        return t["direction"] if t else Direction.UNKNOWN

    def get_summary(self) -> dict:
        up_count   = sum(1 for v in self.violations if v["direction"] == "UP")
        down_count = sum(1 for v in self.violations if v["direction"] == "DOWN")
        return {
            "total_violations": len(self.violations),
            "up_violations":    up_count,
            "down_violations":  down_count,
            "violations":       self.violations,
        }




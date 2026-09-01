"""
violation_rule_engine.py
TVS-9: Violation Rule Engine — Epic 3 final module.

Combines speed estimation (TVS-7) and red light crossing detection (TVS-8)
into a single unified pipeline.

Subtasks:
1. Combine speed and red light checks in a single pipeline
2. Support configurable cooldown per track ID
3. Emit a structured ViolationEvent object with full metadata
4. Unit tests for edge cases

Design:
  - ViolationRuleEngine is a thin coordinator layer. It does NOT
    re-implement detection logic. It subscribes to events emitted
    by SpeedEstimator (TVS-7) and RedLightDetector (TVS-8) and
    merges them into canonical ViolationEvents.
  - One ViolationEvent per (track_id, violation_type) incident.
  - Cooldown is enforced per (track_id, type) pair so a speeding
    vehicle that also runs a red light gets TWO separate events.
  - Evidence clip: the engine records the frame window around each
    violation for downstream plate-crop and OCR modules.
"""

from __future__ import annotations

import cv2
import supervision as sv
from ultralytics import YOLO
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict
import json
import math
import unittest
from datetime import datetime



#  CONFIGURATION


VIDEO_PATH           = "videos/tt.mp4"
MODEL_PATH           = "yolov8n.pt"
VEHICLE_CLASSES      = [2, 3, 5, 7]
CONFIDENCE           = 0.5
IOU                  = 0.45
DISPLAY_WIDTH        = 1280

# --- Speed zone (TVS-7) ---
LINE_UPPER_Y         = 300
LINE_LOWER_Y         = 600
REAL_DISTANCE_METERS = 3.0
VIDEO_FPS            = 30.0
SPEED_LIMIT_KMH      = 60.0
MIN_FRAMES_VALID     = 3

# --- Stop line / Red light (TVS-8) ---
STOP_LINE_Y          = 400

# --- Direction detection ---
DIRECTION_FRAMES     = 3       # Frames to observe before classifying direction
DIRECTION_MIN_MOVE   = 10      # Minimum Y pixel movement for classification

# --- Signal control ---
USE_KEYBOARD         = True
LIGHT_CYCLE_FRAMES   = 300
GREEN_FRAMES         = 120
YELLOW_FRAMES        = 60
RED_FRAMES           = 120

# --- Rule engine ---
COOLDOWN_FRAMES      = 90      # Min frames between two events for same (track, type)
EVIDENCE_PRE_FRAMES  = 15      # Frames before violation to include in clip window
EVIDENCE_POST_FRAMES = 30      # Frames after violation to include in clip window
MIN_TRACK_FRAMES     = 5       # Min frames tracked before red-light check
STALE_TRACK_FRAMES   = 100     # Frames after loss before track cleanup

# --- Ghost / occlusion ---
GHOST_FRAMES         = 10
REASSIGN_DIST        = 100



#  DATA MODEL


class Direction(Enum):
    UNKNOWN = "UNKNOWN"
    UP      = "UP"       # Y decreasing — enters from bottom
    DOWN    = "DOWN"     # Y increasing — enters from top


class SignalState(Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    RED    = "RED"


class ViolationType(str, Enum):
    OVERSPEED  = "OVERSPEED"
    RED_LIGHT  = "RED_LIGHT"


@dataclass
class SpeedMeasurement:
    """Single speed measurement result (TVS-7)."""
    track_id: int
    direction: str
    speed_kmh: float
    violation: bool
    start_frame: int
    end_frame: int
    frames_between: float
    time_seconds: float


@dataclass
class ViolationEvent:
    """
    Canonical violation record emitted by the rule engine.
    Consumed downstream by: plate crop → OCR → MySQL writer → Laravel dashboard.
    """
    event_id:             str              # unique: "{track_id}_{type}_{frame}_{counter}"
    track_id:             int
    violation_type:       ViolationType
    frame_number:         int
    timestamp:            str              # ISO 8601
    direction:            str              # "up" / "down" / "UP" / "DOWN" / "unknown"
    signal_state:         str              # "RED" / "GREEN" / "YELLOW" / "N/A"
    speed_kmh:            Optional[float]  # None for red-light-only events
    speed_limit_kmh:      Optional[float]
    bbox:                 list             # [x1, y1, x2, y2] at violation frame
    evidence_start_frame: int              # clip window start
    evidence_end_frame:   int              # clip window end
    plate_number:         str = ""         # filled by TVS-10/11
    image_path:           str = ""         # filled by TVS-10

    def to_dict(self) -> dict:
        d = asdict(self)
        d["violation_type"] = self.violation_type.value
        return d



#  TRAFFIC LIGHT


class TrafficLight:
    def __init__(self):
        self.state = SignalState.GREEN

    def set_state(self, key: int):
        if key == ord('r'):
            self.state = SignalState.RED
            print("  Signal → RED")
        elif key == ord('y'):
            self.state = SignalState.YELLOW
            print("  Signal → YELLOW")
        elif key == ord('g'):
            self.state = SignalState.GREEN
            print("  Signal → GREEN")

    def auto_update(self, frame_num: int):
        pos = frame_num % LIGHT_CYCLE_FRAMES
        if pos < GREEN_FRAMES:
            self.state = SignalState.GREEN
        elif pos < GREEN_FRAMES + YELLOW_FRAMES:
            self.state = SignalState.YELLOW
        else:
            self.state = SignalState.RED

    def get_bgr(self) -> tuple:
        return {
            SignalState.GREEN:  (0, 255, 0),
            SignalState.YELLOW: (0, 255, 255),
            SignalState.RED:    (0, 0, 255),
        }[self.state]



#  TVS-7: SPEED ESTIMATOR
#  History buffer + sub-frame interpolation for high-speed accuracy.


class SpeedEstimator:
    """
    Advanced speed estimator with a Coordinate History Buffer
    and Sub-Frame Interpolation for high-speed accuracy.
    Ported from TVS-7 (speed_estimation_bidirectional.py).
    """

    def __init__(self, line_upper: int, line_lower: int,
                 real_distance_m: float, fps: float, speed_limit: float):
        self.line_upper       = line_upper
        self.line_lower       = line_lower
        self.real_distance_m  = real_distance_m
        self.fps              = fps
        self.speed_limit      = speed_limit
        self.pixel_distance   = abs(line_lower - line_upper)
        self.pixels_per_meter = self.pixel_distance / real_distance_m

        self._tracks: Dict[int, dict] = {}
        self._ghosts: List[dict]      = []
        self.measurements: List[SpeedMeasurement] = []
        self.discarded_count = 0

    def _get_track(self, track_id: int) -> dict:
        """Get or create track state using a history buffer."""
        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "state": "active",
                "direction": None,
                "history": [],  # (frame_num, top_y, bottom_y, center_y)
            }
        return self._tracks[track_id]

    def _find_ghost_match(self, cx: float, cy: float) -> Optional[dict]:
        best_match = None
        best_dist = float('inf')
        for ghost in self._ghosts:
            if ghost["frames_since_lost"] > GHOST_FRAMES:
                continue
            gx, gy = ghost["last_pos"]
            dist = math.sqrt((cx - gx)**2 + (cy - gy)**2)
            if dist < REASSIGN_DIST and dist < best_dist:
                best_dist = dist
                best_match = ghost
        return best_match

    def _ghost_track(self, track_id: int, track: dict):
        if len(track["history"]) > 0:
            last_cx_cy = (0, track["history"][-1][3])
            self._ghosts.append({
                "last_pos": last_cx_cy,
                "state": track.copy(),
                "frames_since_lost": 0,
            })

    def process(self, detections: sv.Detections, frame_num: int) -> List[dict]:
        """
        Process detections for speed estimation.
        Returns list of speed event dicts (with bbox for rule engine).
        """
        events = []

        if detections.tracker_id is None:
            for tid, track in list(self._tracks.items()):
                self._ghost_track(tid, track)
            self._tracks.clear()
            return events

        tracker_ids = detections.tracker_id
        current_ids = set(tracker_ids.tolist() if hasattr(tracker_ids, 'tolist') else tracker_ids)
        lost_ids = set(self._tracks.keys()) - current_ids

        for tid in lost_ids:
            self._ghost_track(tid, self._tracks[tid])
            del self._tracks[tid]

        tracker_ids = detections.tracker_id
        for i, track_id in enumerate(tracker_ids.tolist() if hasattr(tracker_ids, 'tolist') else tracker_ids):
            x1, y1, x2, y2 = detections.xyxy[i]
            top_y, bottom_y = float(y1), float(y2)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if track_id not in self._tracks:
                ghost = self._find_ghost_match(cx, cy)
                if ghost:
                    self._tracks[track_id] = ghost["state"].copy()
                    self._ghosts.remove(ghost)
                else:
                    self._get_track(track_id)

            track = self._tracks[track_id]

            # 1. ADD TO HISTORY BUFFER
            if track["state"] == "active":
                track["history"].append((frame_num, top_y, bottom_y, cy))

                # 2. EVALUATE THE BUFFER
                event = self._evaluate_history(
                    track, track_id,
                    bbox=[float(x1), float(y1), float(x2), float(y2)]
                )
                if event:
                    events.append(event)

        # Clean old ghosts
        self._ghosts = [g for g in self._ghosts if g["frames_since_lost"] <= GHOST_FRAMES]
        for g in self._ghosts:
            g["frames_since_lost"] += 1

        return events

    def _get_exact_crossing_frame(self, history: list, line_y: float,
                                   is_up: bool, edge_idx: int) -> Optional[float]:
        """Calculates the exact sub-frame a line was crossed using linear interpolation."""
        for i in range(1, len(history)):
            prev_f = history[i-1][0]
            curr_f = history[i][0]

            y_prev = history[i-1][edge_idx]
            y_curr = history[i][edge_idx]

            if is_up:  # Moving up: Y is decreasing
                if y_prev > line_y >= y_curr:
                    ratio = (y_prev - line_y) / (y_prev - y_curr + 1e-6)
                    return prev_f + ratio * (curr_f - prev_f)
            else:      # Moving down: Y is increasing
                if y_prev < line_y <= y_curr:
                    ratio = (line_y - y_prev) / (y_curr - y_prev + 1e-6)
                    return prev_f + ratio * (curr_f - prev_f)
        return None

    def _evaluate_history(self, track: dict, track_id: int,
                          bbox: list = None) -> Optional[dict]:
        history = track["history"]

        # --- Step A: Determine Direction ---
        if track["direction"] is None:
            if len(history) >= DIRECTION_FRAMES:
                first_y = history[0][3]
                last_y = history[-1][3]
                diff = last_y - first_y

                if abs(diff) < 5:
                    if len(history) > 90:
                        track["state"] = "discarded"
                        self.discarded_count += 1
                    return None

                track["direction"] = "up" if diff < 0 else "down"
            return None

        # --- Step B: Check History Buffer for Crossings ---
        direction = track["direction"]

        if direction == "up":
            # Edge index 1 is top_y
            start_frame_exact = self._get_exact_crossing_frame(
                history, self.line_lower, True, 1)
            end_frame_exact = self._get_exact_crossing_frame(
                history, self.line_upper, True, 1)
        else:
            # Edge index 2 is bottom_y
            start_frame_exact = self._get_exact_crossing_frame(
                history, self.line_upper, False, 2)
            end_frame_exact = self._get_exact_crossing_frame(
                history, self.line_lower, False, 2)

        # --- Step C: Calculate Speed ---
        if start_frame_exact is not None and end_frame_exact is not None:
            frames = end_frame_exact - start_frame_exact

            if frames <= 0 or frames < MIN_FRAMES_VALID:
                track["state"] = "discarded"
                self.discarded_count += 1
                return None

            time_s = frames / self.fps
            speed_ms = self.real_distance_m / time_s
            speed_kmh = round(speed_ms * 3.6, 1)
            violation = speed_kmh > self.speed_limit

            measurement = SpeedMeasurement(
                track_id=track_id,
                direction=direction,
                speed_kmh=speed_kmh,
                violation=violation,
                start_frame=int(start_frame_exact),
                end_frame=int(end_frame_exact),
                frames_between=round(frames, 2),
                time_seconds=round(time_s, 3),
            )

            self.measurements.append(measurement)
            track["state"] = "done"
            track["history"] = []  # Clear memory

            return {
                "track_id":   track_id,
                "direction":  direction,
                "speed_kmh":  speed_kmh,
                "violation":  violation,
                "frames":     round(frames, 2),
                "time_s":     round(time_s, 3),
                "frame_num":  int(end_frame_exact),
                "bbox":       bbox or [0, 0, 0, 0],
            }

        return None

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_speed(self, track_id: int) -> Optional[float]:
        for m in self.measurements:
            if m.track_id == track_id:
                return m.speed_kmh
        return None

    def is_violation(self, track_id: int) -> bool:
        for m in self.measurements:
            if m.track_id == track_id:
                return m.violation
        return False

    def get_track_state(self, track_id: int) -> str:
        if track_id in self._tracks:
            return self._tracks[track_id]["state"]
        return "unknown"

    def get_direction(self, track_id: int) -> Optional[str]:
        if track_id in self._tracks:
            return self._tracks[track_id].get("direction")
        return None

    def get_summary(self) -> dict:
        total = len(self.measurements)
        violations = sum(1 for m in self.measurements if m.violation)
        avg_speed = sum(m.speed_kmh for m in self.measurements) / total if total else 0
        up = sum(1 for m in self.measurements if m.direction == "up")
        down = sum(1 for m in self.measurements if m.direction == "down")

        return {
            "total_valid": total,
            "up_count": up,
            "down_count": down,
            "discarded": self.discarded_count,
            "violations": violations,
            "average_speed": round(avg_speed, 1),
            "speed_limit": self.speed_limit,
            "real_distance_m": self.real_distance_m,
            "pixels_per_meter": round(self.pixels_per_meter, 1),
            "measurements": [
                {
                    "track_id": m.track_id,
                    "direction": m.direction,
                    "speed_kmh": m.speed_kmh,
                    "violation": m.violation,
                    "frames": m.frames_between,
                    "time_s": m.time_seconds,
                }
                for m in self.measurements
            ]
        }



#  TVS-8: RED LIGHT DETECTOR
#  Bidirectional, history buffer, retroactive check, signal-per-frame fix.


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



#  OCCLUSION TRACKER
#  Consistent ID reassignment via ghost tracking.


class OcclusionTracker:
    def __init__(self, frame_rate=30):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=frame_rate,
        )
        self.ghost_tracks       = {}
        self.id_map             = {}
        self.next_consistent_id = 1
        self.active_ids         = set()

    def update(self, detections, frame_num):
        tracked     = self.tracker.update_with_detections(detections)
        prev_active = self.active_ids.copy()
        self.active_ids = set()

        if tracked.tracker_id is None:
            return tracked, prev_active - self.active_ids

        new_ids = []
        for i, tid in enumerate(tracked.tracker_id.tolist()):
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if tid in self.id_map:
                cid = self.id_map[tid]
            else:
                cid = self._try_reassign(cx, cy, frame_num)
                if cid is None:
                    cid = self.next_consistent_id
                    self.next_consistent_id += 1
                self.id_map[tid] = cid

            new_ids.append(cid)
            self.active_ids.add(cid)
            self.ghost_tracks[cid] = {
                "last_pos":   (float(cx), float(cy)),
                "lost_frame": int(frame_num),
            }

        self._clean_ghosts(frame_num)
        tracked.tracker_id = new_ids
        return tracked, prev_active - self.active_ids

    def _try_reassign(self, cx, cy, frame_num):
        best, best_dist = None, float("inf")
        for gid, g in self.ghost_tracks.items():
            if frame_num - g["lost_frame"] > 15:
                continue
            d = math.sqrt((float(cx) - g["last_pos"][0]) ** 2 +
                          (float(cy) - g["last_pos"][1]) ** 2)
            if d < 80 and d < best_dist:
                best_dist = d
                best      = gid
        if best is not None:
            del self.ghost_tracks[best]
            return best
        return None

    def _clean_ghosts(self, frame_num):
        stale = [gid for gid, g in self.ghost_tracks.items()
                 if frame_num - g["lost_frame"] > 15]
        for gid in stale:
            del self.ghost_tracks[gid]



#  TVS-9: VIOLATION RULE ENGINE
#  Thin coordinator — owns both sub-detectors, applies cooldown,
#  emits canonical ViolationEvent objects.


class ViolationRuleEngine:
    """
    TVS-9: Combines speed (TVS-7) and red light (TVS-8) detectors.

    Responsibilities:
      1. Receive raw events from both detectors each frame
      2. Apply per-(track, type) cooldown to prevent duplicate events
      3. Emit canonical ViolationEvent objects with full metadata
      4. Maintain a running log for downstream modules (plate crop, OCR, DB)

    The engine owns NO detection logic — it only coordinates and enriches.
    """

    def __init__(self,
                 speed_estimator:    SpeedEstimator,
                 red_light_detector: RedLightDetector,
                 speed_limit_kmh:    float,
                 cooldown_frames:    int = COOLDOWN_FRAMES,
                 evidence_pre:       int = EVIDENCE_PRE_FRAMES,
                 evidence_post:      int = EVIDENCE_POST_FRAMES):

        self.speed_est    = speed_estimator
        self.rl_detector  = red_light_detector
        self.speed_limit  = speed_limit_kmh
        self.cooldown     = cooldown_frames
        self.evidence_pre = evidence_pre
        self.evidence_post = evidence_post

        # Cooldown tracker: {(track_id, ViolationType): last_triggered_frame}
        self._last_triggered: Dict[tuple, int] = {}

        # All emitted events this session
        self.events: List[ViolationEvent] = []
        self._event_counter = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_on_cooldown(self, tid: int, vtype: ViolationType,
                        frame_num: int) -> bool:
        key  = (tid, vtype)
        last = self._last_triggered.get(key, -999999)
        return (frame_num - last) <= self.cooldown

    def _mark_triggered(self, tid: int, vtype: ViolationType,
                        frame_num: int):
        self._last_triggered[(tid, vtype)] = frame_num

    def _make_event_id(self, tid: int, vtype: ViolationType,
                       frame_num: int) -> str:
        self._event_counter += 1
        return f"{tid}_{vtype.value}_{frame_num}_{self._event_counter}"

    def _emit(self,
              tid:        int,
              vtype:      ViolationType,
              frame_num:  int,
              direction:  str,
              signal:     str,
              speed_kmh:  Optional[float],
              bbox:       list) -> ViolationEvent:

        event = ViolationEvent(
            event_id             = self._make_event_id(tid, vtype, frame_num),
            track_id             = tid,
            violation_type       = vtype,
            frame_number         = frame_num,
            timestamp            = datetime.now().isoformat(),
            direction            = direction,
            signal_state         = signal,
            speed_kmh            = speed_kmh,
            speed_limit_kmh      = self.speed_limit if vtype == ViolationType.OVERSPEED else None,
            bbox                 = [round(v, 1) for v in bbox],
            evidence_start_frame = max(0, frame_num - self.evidence_pre),
            evidence_end_frame   = frame_num + self.evidence_post,
        )
        self.events.append(event)
        self._mark_triggered(tid, vtype, frame_num)
        return event

    # ── Main per-frame update ─────────────────────────────────────────────────

    def update(self,
               detections: sv.Detections,
               signal:     SignalState,
               frame_num:  int) -> List[ViolationEvent]:
        """
        Call once per frame with the current detections and signal state.
        Returns list of new ViolationEvents emitted this frame.
        """
        new_events: List[ViolationEvent] = []

        # --- Run sub-detectors ---
        speed_events = self.speed_est.process(detections, frame_num)
        rl_events    = self.rl_detector.process(detections, signal, frame_num)

        # --- Process speed violations ---
        for se in speed_events:
            if not se["violation"]:
                continue
            tid   = se["track_id"]
            vtype = ViolationType.OVERSPEED
            if self._is_on_cooldown(tid, vtype, frame_num):
                continue
            ev = self._emit(
                tid       = tid,
                vtype     = vtype,
                frame_num = se.get("frame_num", frame_num),
                direction = se["direction"],
                signal    = signal.value,
                speed_kmh = se["speed_kmh"],
                bbox      = se.get("bbox", [0, 0, 0, 0]),
            )
            new_events.append(ev)
            print(f"  [RULE ENGINE] Frame {frame_num}: OVERSPEED — "
                  f"Vehicle #{tid} @ {se['speed_kmh']} km/h  "
                  f"[event_id={ev.event_id}]")

        # --- Process red light violations ---
        for rle in rl_events:
            tid   = rle["track_id"]
            vtype = ViolationType.RED_LIGHT
            if self._is_on_cooldown(tid, vtype, frame_num):
                continue

            # Enrich with speed if we happen to have it
            speed = self.speed_est.get_speed(tid)

            ev = self._emit(
                tid       = tid,
                vtype     = vtype,
                frame_num = rle["frame"],
                direction = rle.get("direction", "unknown"),
                signal    = rle.get("signal_state", "RED"),
                speed_kmh = speed,
                bbox      = rle.get("bbox", [0, 0, 0, 0]),
            )
            new_events.append(ev)
            print(f"  [RULE ENGINE] Frame {frame_num}: RED_LIGHT — "
                  f"Vehicle #{tid} direction={rle.get('direction', '?')}  "
                  f"[event_id={ev.event_id}]")

        return new_events

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_events_for_track(self, tid: int) -> List[ViolationEvent]:
        return [e for e in self.events if e.track_id == tid]

    def get_events_by_type(self, vtype: ViolationType) -> List[ViolationEvent]:
        return [e for e in self.events if e.violation_type == vtype]

    def get_summary(self) -> dict:
        overspeed = self.get_events_by_type(ViolationType.OVERSPEED)
        red_light = self.get_events_by_type(ViolationType.RED_LIGHT)
        combined  = [tid for tid in {e.track_id for e in overspeed}
                     if any(e.track_id == tid for e in red_light)]
        return {
            "total_events":    len(self.events),
            "overspeed_count": len(overspeed),
            "red_light_count": len(red_light),
            "combined_count":  len(combined),
            "unique_vehicles": len({e.track_id for e in self.events}),
            "cooldown_frames": self.cooldown,
            "speed_limit_kmh": self.speed_limit,
            "events": [e.to_dict() for e in self.events],
        }



#  DRAWING / HUD


DIR_COLORS = {
    Direction.UP:      (255, 200, 0),     # cyan-ish
    Direction.DOWN:    (0, 165, 255),      # orange
    Direction.UNKNOWN: (160, 160, 160),    # grey
}
DIR_ARROWS = {
    Direction.UP:      "↑",
    Direction.DOWN:    "↓",
    Direction.UNKNOWN: "?",
}


def draw_hud(frame, frame_num: int, engine: ViolationRuleEngine,
             tracked, signal: SignalState, orig_w: int, orig_h: int):
    """Draw combined overlays: speed zone + stop line + traffic light + summary."""

    # Speed zone shading
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, LINE_UPPER_Y), (orig_w, LINE_LOWER_Y),
                  (255, 255, 0), -1)
    frame[:] = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

    # Speed lines
    cv2.line(frame, (0, LINE_UPPER_Y), (orig_w, LINE_UPPER_Y), (0, 0, 255), 2)
    cv2.putText(frame, f"UPPER LINE (Y={LINE_UPPER_Y})",
                (10, LINE_UPPER_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    cv2.line(frame, (0, LINE_LOWER_Y), (orig_w, LINE_LOWER_Y), (0, 255, 0), 2)
    cv2.putText(frame, f"LOWER LINE (Y={LINE_LOWER_Y})",
                (10, LINE_LOWER_Y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # Stop line
    sl_color     = (0, 0, 255) if signal == SignalState.RED else (180, 180, 180)
    sl_thickness = 4           if signal == SignalState.RED else 2
    cv2.line(frame, (0, STOP_LINE_Y), (orig_w, STOP_LINE_Y), sl_color, sl_thickness)
    cv2.putText(frame, f"STOP LINE (Y={STOP_LINE_Y})",
                (10, STOP_LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, sl_color, 2)

    # Direction indicators
    mid_y = (LINE_UPPER_Y + LINE_LOWER_Y) // 2
    cv2.putText(frame, "UP "+chr(8593), (50, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, chr(8595)+" DOWN", (orig_w - 180, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Traffic light widget
    lx, ly = orig_w - 110, 30
    cv2.rectangle(frame, (lx - 15, ly - 10), (lx + 75, ly + 110), (40, 40, 40), -1)
    cv2.rectangle(frame, (lx - 15, ly - 10), (lx + 75, ly + 110), (180, 180, 180), 2)
    for idx, st in enumerate([SignalState.RED, SignalState.YELLOW, SignalState.GREEN]):
        cy_ = ly + 15 + idx * 32
        col = {SignalState.RED: (0,0,255), SignalState.YELLOW: (0,255,255),
               SignalState.GREEN: (0,255,0)}[st]
        if st == signal:
            cv2.circle(frame, (lx + 30, cy_), 13, col, -1)
            cv2.circle(frame, (lx + 30, cy_), 13, (255, 255, 255), 2)
        else:
            cv2.circle(frame, (lx + 30, cy_), 13, (50, 50, 50), -1)
    cv2.putText(frame, signal.value, (lx - 10, ly + 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                {SignalState.GREEN: (0,255,0), SignalState.YELLOW: (0,255,255),
                 SignalState.RED: (0,0,255)}[signal], 2)

    if USE_KEYBOARD:
        cv2.putText(frame, "R/Y/G=signal  Q=quit  P=pause",
                    (10, orig_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Violation flash overlay for recent events
    recent = [e for e in engine.events
              if frame_num - e.frame_number <= EVIDENCE_POST_FRAMES]
    if recent:
        flash_overlay = frame.copy()
        cv2.rectangle(flash_overlay, (0, 0), (orig_w, orig_h), (0, 0, 180), -1)
        frame[:] = cv2.addWeighted(frame, 0.92, flash_overlay, 0.08, 0)

    # Summary box
    summary = engine.get_summary()
    cv2.rectangle(frame, (8, 8), (460, 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (8, 8), (460, 80), (80, 80, 80), 1)
    active = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
    cv2.putText(frame,
                f"Frame:{frame_num} | Active:{active} | Signal:{signal.value}",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(frame,
                f"Violations: {summary['total_events']}  "
                f"(Speed:{summary['overspeed_count']}  "
                f"Red:{summary['red_light_count']}  "
                f"Both:{summary['combined_count']})",
                (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    cv2.putText(frame,
                f"Unique vehicles: {summary['unique_vehicles']}",
                (14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def draw_direction_indicators(frame, tracked, rl_detector: RedLightDetector):
    """Draw a small colored arrow on each vehicle box indicating detected direction."""
    if tracked.tracker_id is None:
        return
    for i, track_id in enumerate(tracked.tracker_id):
        direction = rl_detector.get_direction(track_id)
        color     = DIR_COLORS[direction]
        x1, y1, x2, y2 = tracked.xyxy[i]
        cx = int((x1 + x2) / 2)
        if direction == Direction.UP:
            cv2.arrowedLine(frame, (cx, int(y1) + 20), (cx, int(y1) - 5),
                            color, 2, tipLength=0.4)
        elif direction == Direction.DOWN:
            cv2.arrowedLine(frame, (cx, int(y1) - 5), (cx, int(y1) + 20),
                            color, 2, tipLength=0.4)


def build_labels(tracked, engine: ViolationRuleEngine, model) -> list:
    labels = []
    if tracked.tracker_id is None:
        return labels
    for class_id, tid in zip(tracked.class_id, tracked.tracker_id):
        name      = model.names[class_id]
        speed     = engine.speed_est.get_speed(tid)
        direction = engine.speed_est.get_direction(tid)
        arrow     = "↑" if direction == "up" else "↓" if direction == "down" else "?"
        evs       = engine.get_events_for_track(tid)
        types     = {e.violation_type for e in evs}

        parts = [f"#{tid}", arrow, name]
        if speed is not None:
            parts.append(f"{speed}km/h")
        if ViolationType.OVERSPEED in types:
            parts.append("[!SPEED]")
        if ViolationType.RED_LIGHT in types:
            parts.append("[!RED]")
        labels.append(" ".join(parts))
    return labels


def resize_for_display(frame, target_width=1280):
    h, w = frame.shape[:2]
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)),
                      interpolation=cv2.INTER_AREA)



#  UNIT TESTS


class TestViolationRuleEngine(unittest.TestCase):

    def _make_engine(self, cooldown=COOLDOWN_FRAMES):
        se  = SpeedEstimator(LINE_UPPER_Y, LINE_LOWER_Y,
                             REAL_DISTANCE_METERS, VIDEO_FPS, SPEED_LIMIT_KMH)
        rld = RedLightDetector(STOP_LINE_Y)
        return ViolationRuleEngine(se, rld, SPEED_LIMIT_KMH,
                                   cooldown_frames=cooldown)

    def _make_det(self, tid, x1, y1, x2, y2):
        """Build a minimal sv.Detections mock."""
        import numpy as np
        d = sv.Detections(
            xyxy       = np.array([[x1, y1, x2, y2]], dtype=float),
            confidence = np.array([0.9]),
            class_id   = np.array([2]),
            tracker_id = np.array([tid]),
        )
        return d

    def test_cooldown_prevents_duplicate(self):
        """Same (track, type) within cooldown window must not emit twice."""
        engine = self._make_engine()
        engine._emit(1, ViolationType.OVERSPEED, 100, "up", "GREEN", 80.0,
                     [0, 0, 50, 50])
        # Frame 110 — within cooldown (90 frames)
        on_cd = engine._is_on_cooldown(1, ViolationType.OVERSPEED, 110)
        self.assertTrue(on_cd)
        self.assertEqual(len(engine.events), 1)

    def test_cooldown_expires(self):
        """After cooldown, same track can emit again."""
        engine = self._make_engine()
        engine._emit(1, ViolationType.OVERSPEED, 100, "up", "GREEN", 80.0,
                     [0, 0, 50, 50])
        # Frame 100 + 91 = 191 — past cooldown
        on_cd = engine._is_on_cooldown(1, ViolationType.OVERSPEED, 191)
        self.assertFalse(on_cd)

    def test_different_types_independent_cooldown(self):
        """OVERSPEED and RED_LIGHT cooldowns are independent per track."""
        engine = self._make_engine()
        engine._emit(1, ViolationType.OVERSPEED, 100, "up", "GREEN", 80.0,
                     [0, 0, 50, 50])
        engine._emit(1, ViolationType.RED_LIGHT, 100, "up", "RED", None,
                     [0, 0, 50, 50])
        self.assertEqual(len(engine.events), 2)
        # Both are on cooldown individually
        self.assertTrue(engine._is_on_cooldown(1, ViolationType.OVERSPEED, 150))
        self.assertTrue(engine._is_on_cooldown(1, ViolationType.RED_LIGHT, 150))

    def test_different_tracks_independent(self):
        """Cooldown on track 1 must not affect track 2."""
        engine = self._make_engine()
        engine._emit(1, ViolationType.OVERSPEED, 100, "up", "GREEN", 80.0,
                     [0, 0, 50, 50])
        on_cd = engine._is_on_cooldown(2, ViolationType.OVERSPEED, 110)
        self.assertFalse(on_cd)

    def test_event_fields(self):
        """ViolationEvent must carry all required fields for downstream modules."""
        engine = self._make_engine()
        ev = engine._emit(5, ViolationType.RED_LIGHT, 200, "down", "RED",
                          None, [10, 20, 60, 80])
        self.assertEqual(ev.track_id, 5)
        self.assertEqual(ev.violation_type, ViolationType.RED_LIGHT)
        self.assertEqual(ev.frame_number, 200)
        self.assertEqual(ev.evidence_start_frame, 200 - EVIDENCE_PRE_FRAMES)
        self.assertEqual(ev.evidence_end_frame, 200 + EVIDENCE_POST_FRAMES)
        self.assertIsNone(ev.speed_kmh)
        self.assertEqual(ev.plate_number, "")  # unfilled until TVS-10

    def test_event_id_unique(self):
        """Every emitted event must have a unique event_id."""
        engine = self._make_engine()
        ids = [
            engine._emit(i, ViolationType.OVERSPEED, 100 + i, "up", "G",
                         80.0, [0, 0, 1, 1]).event_id
            for i in range(10)
        ]
        self.assertEqual(len(ids), len(set(ids)))

    def test_summary_counts(self):
        """Summary must correctly count by type and unique vehicles."""
        engine = self._make_engine()
        engine._emit(1, ViolationType.OVERSPEED, 100, "up", "G", 80.0,
                     [0, 0, 1, 1])
        engine._emit(2, ViolationType.RED_LIGHT, 200, "up", "R", None,
                     [0, 0, 1, 1])
        engine._emit(3, ViolationType.OVERSPEED, 300, "up", "G", 90.0,
                     [0, 0, 1, 1])
        engine._emit(3, ViolationType.RED_LIGHT, 300, "up", "R", 90.0,
                     [0, 0, 1, 1])
        s = engine.get_summary()
        self.assertEqual(s["total_events"],    4)
        self.assertEqual(s["overspeed_count"], 2)
        self.assertEqual(s["red_light_count"], 2)
        self.assertEqual(s["combined_count"],  1)   # vehicle #3
        self.assertEqual(s["unique_vehicles"], 3)

    def test_to_dict_serializable(self):
        """ViolationEvent.to_dict() must produce JSON-serializable output."""
        engine = self._make_engine()
        ev = engine._emit(1, ViolationType.OVERSPEED, 100, "up", "G", 80.0,
                          [0, 0, 50, 80])
        d = ev.to_dict()
        json.dumps(d)  # must not raise



#  MAIN


def main():
    import sys
    if "--test" in sys.argv:
        print("Running TVS-9 unit tests...")
        suite  = unittest.TestLoader().loadTestsFromTestCase(TestViolationRuleEngine)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1

    print("=" * 60)
    print("TVS-9: Violation Rule Engine")
    print("  Speed Estimation (TVS-7) + Red Light Detection (TVS-8)")
    print("=" * 60)
    print(f"Speed zone    : Y={LINE_UPPER_Y} (upper) to Y={LINE_LOWER_Y} (lower)")
    print(f"Real distance : {REAL_DISTANCE_METERS} m")
    print(f"Speed limit   : {SPEED_LIMIT_KMH} km/h")
    print(f"Stop line     : Y={STOP_LINE_Y}")
    print(f"Cooldown      : {COOLDOWN_FRAMES} frames")
    print(f"Evidence clip : -{EVIDENCE_PRE_FRAMES} / +{EVIDENCE_POST_FRAMES} frames")
    print(f"Signal mode   : {'KEYBOARD (R/Y/G)' if USE_KEYBOARD else 'AUTO CYCLE'}")
    print(f"FPS           : {VIDEO_FPS}")
    print("-" * 60)

    model   = YOLO(MODEL_PATH)
    tracker = OcclusionTracker(frame_rate=VIDEO_FPS)
    light   = TrafficLight()

    speed_est = SpeedEstimator(
        LINE_UPPER_Y, LINE_LOWER_Y,
        REAL_DISTANCE_METERS, VIDEO_FPS, SPEED_LIMIT_KMH
    )
    rl_det = RedLightDetector(STOP_LINE_Y)
    engine = ViolationRuleEngine(speed_est, rl_det, SPEED_LIMIT_KMH)

    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.55)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open video:", VIDEO_PATH)
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video         : {orig_w}x{orig_h}")
    print("Controls      : Q=quit  P=pause  R=Red  Y=Yellow  G=Green")
    print("=" * 60)

    frame_count = 0
    paused      = False

    while True:
        # Key read FIRST — used for signal update this same frame
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print("  [PAUSED]" if paused else "  [RESUMED]")

        if USE_KEYBOARD:
            if key in (ord('r'), ord('y'), ord('g')):
                light.set_state(key)
        else:
            if not paused:
                light.auto_update(frame_count)

        if paused:
            continue

        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Detection & tracking
        results    = model(frame, classes=VEHICLE_CLASSES,
                           conf=CONFIDENCE, iou=IOU, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        tracked, _ = tracker.update(detections, frame_count)

        # ── Single call to the rule engine ────────────────────────────────
        new_events = engine.update(tracked, light.state, frame_count)

        # Draw scene
        annotated = frame.copy()
        draw_hud(annotated, frame_count, engine, tracked,
                 light.state, orig_w, orig_h)
        draw_direction_indicators(annotated, tracked, engine.rl_detector)

        labels    = build_labels(tracked, engine, model)
        annotated = box_annotator.annotate(scene=annotated, detections=tracked)
        annotated = label_annotator.annotate(scene=annotated,
                                              detections=tracked,
                                              labels=labels)

        display = resize_for_display(annotated, DISPLAY_WIDTH)
        cv2.imshow("TVS-9 Violation Rule Engine", display)

    cap.release()
    cv2.destroyAllWindows()

    # ── Final Report ──────────────────────────────────────────────────────
    summary = engine.get_summary()
    speed_summary = engine.speed_est.get_summary()
    rl_summary    = engine.rl_detector.get_summary()

    print("\n" + "=" * 60)
    print("TVS-9 VIOLATION RULE ENGINE — FINAL REPORT")
    print("=" * 60)
    print(f"Total events      : {summary['total_events']}")
    print(f"  Overspeed       : {summary['overspeed_count']}")
    print(f"  Red light       : {summary['red_light_count']}")
    print(f"  Both types      : {summary['combined_count']}")
    print(f"Unique vehicles   : {summary['unique_vehicles']}")

    print(f"\nSpeed measurements: {speed_summary['total_valid']} "
          f"(UP: {speed_summary['up_count']}, DOWN: {speed_summary['down_count']})")
    print(f"  Discarded       : {speed_summary['discarded']}")
    print(f"  Average speed   : {speed_summary['average_speed']} km/h")
    print(f"  Calibration     : {speed_summary['pixels_per_meter']} px/m")

    print(f"\nRed light hits    : {rl_summary['total_violations']} "
          f"(UP: {rl_summary['up_violations']}, DOWN: {rl_summary['down_violations']})")

    if summary["events"]:
        print("\nEvent log:")
        for e in summary["events"]:
            speed_str = f"{e['speed_kmh']} km/h" if e["speed_kmh"] else "N/A"
            print(f"  [{e['event_id']}]  "
                  f"Frame {e['frame_number']:>5}  "
                  f"Vehicle #{e['track_id']:>3}  "
                  f"{e['violation_type']:<10}  "
                  f"speed={speed_str:<12}  "
                  f"dir={e['direction']:<5}  "
                  f"signal={e['signal_state']}")
            print(f"           evidence frames: "
                  f"{e['evidence_start_frame']} -> {e['evidence_end_frame']}")

    # Save combined JSON report
    combined_report = {
        "rule_engine": summary,
        "speed_estimation": speed_summary,
        "red_light_detection": rl_summary,
    }
    out = "violation_events.json"
    with open(out, "w") as f:
        json.dump(combined_report, f, indent=2)
    print(f"\nSaved: {out}")
    print("Run with --test flag to execute unit tests.")
    print("=" * 60)


if __name__ == "__main__":
    main()

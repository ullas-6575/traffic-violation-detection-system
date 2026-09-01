"""TVS-7: Bidirectional speed estimation with sub-frame interpolation."""

from __future__ import annotations

import math

import supervision as sv

from .config import DIRECTION_FRAMES, MIN_FRAMES_VALID
from .models import Direction, SpeedMeasurement


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




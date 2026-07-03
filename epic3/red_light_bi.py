"""
red_light_detection.py
TVS-8: Red light crossing detection — BIDIRECTIONAL traffic support.

Direction is auto-detected per track using Y movement over first N frames.
  UP   vehicles (Y decreasing) → top edge (y1) crosses stop line going up
  DOWN vehicles (Y increasing) → bottom edge (y2) crosses stop line going down
"""

import cv2
import supervision as sv
from ultralytics import YOLO
import math
import json
from datetime import datetime
from enum import Enum
from typing import Optional 

# ========== CONFIGURATION ==========
VIDEO_PATH      = "videos/tt.mp4"
MODEL_PATH      = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE      = 0.5
IOU             = 0.45
DISPLAY_WIDTH   = 1280

# --- STOP LINE ---
STOP_LINE_Y = 400

# --- DIRECTION DETECTION ---
# How many frames to observe before classifying direction
DIRECTION_FRAMES   = 8
# Minimum Y pixel movement to be classified (filters stationary/noise)
DIRECTION_MIN_MOVE = 10

# --- SIGNAL CONTROL ---
USE_KEYBOARD       = True
LIGHT_CYCLE_FRAMES = 300
GREEN_FRAMES       = 120
YELLOW_FRAMES      = 60
RED_FRAMES         = 120

# --- VIOLATION SETTINGS ---
VIOLATION_COOLDOWN_FRAMES = 90
MIN_TRACK_FRAMES          = 5
STALE_TRACK_FRAMES        = 100
# ==================================


class Direction(Enum):
    UNKNOWN = "UNKNOWN"
    UP      = "UP"      # Y decreasing — enters from bottom
    DOWN    = "DOWN"    # Y increasing — enters from top


class SignalState(Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    RED    = "RED"


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


class BidirectionalRedLightDetector:
    """
    Red light violation detector for bidirectional traffic.
    UPGRADED: Uses a Coordinate History Buffer to catch fast vehicles
    that cross the line before their direction is classified.
    """

    def __init__(self, stop_line_y: int):
        self.stop_line_y = stop_line_y
        self.tracks      = {}
        self.violations  = []

    # ── Track management ──────────────────────────────────────────────────────

    def _init_track(self, track_id: int, frame_num: int):
        self.tracks[track_id] = {
            "direction":      Direction.UNKNOWN,
            "history":        [], # Replaced y_samples with full history buffer
            "last_edge_y":    None,
            "frames_tracked": 0,
            "last_seen_frame": frame_num,
            "violated":       False,
            "violation_frame": None,
        }

    def _classify_direction(self, track: dict) -> Direction:
        """
        Compare first and last Y sample in the history buffer.
        Negative delta → moving UP. Positive delta → moving DOWN.
        """
        history = track["history"]
        if len(history) < DIRECTION_FRAMES:
            return Direction.UNKNOWN

        # History tuples are: (frame_num, top_y, bottom_y, center_y)
        # Index 3 is center_y
        delta = history[-1][3] - history[0][3]
        if abs(delta) < DIRECTION_MIN_MOVE:
            return Direction.UNKNOWN   # stationary or noise
        return Direction.UP if delta < 0 else Direction.DOWN

    def _check_retroactive_violation(self, track: dict, signal: SignalState) -> Optional[dict]:
        """
        Scans the history buffer. If a vehicle ran the red light DURING 
        the classification blind spot, it catches them retroactively.
        """
        if signal != SignalState.RED:
            return None
            
        direction = track["direction"]
        history = track["history"]
        
        # Scan through the buffer for line crossings
        for i in range(1, len(history)):
            prev_frame, prev_top, prev_bottom, _ = history[i-1]
            curr_frame, curr_top, curr_bottom, _ = history[i]
            
            if direction == Direction.UP:
                if prev_top > self.stop_line_y >= curr_top:
                    return {"frame": curr_frame, "edge_label": "top_y", "edge_val": curr_top}
            elif direction == Direction.DOWN:
                if prev_bottom < self.stop_line_y <= curr_bottom:
                    return {"frame": curr_frame, "edge_label": "bottom_y", "edge_val": curr_bottom}
        return None

    # ── Main update ───────────────────────────────────────────────────────────

    def process(self, detections: sv.Detections, signal: SignalState, frame_num: int):
        events     = []
        active_ids = set()

        if detections.tracker_id is None:
            self._cleanup(frame_num, active_ids)
            return events

        active_ids = set(int(tid) for tid in detections.tracker_id)

        for i, track_id in enumerate(detections.tracker_id):
            track_id        = int(track_id)
            x1, y1, x2, y2 = detections.xyxy[i]
            top_y           = float(y1)
            bottom_y        = float(y2)
            center_y        = (top_y + bottom_y) / 2

            # Init new track
            if track_id not in self.tracks:
                self._init_track(track_id, frame_num)

            track = self.tracks[track_id]
            track["frames_tracked"]   += 1
            track["last_seen_frame"]   = frame_num

            # 1. ALWAYS ADD TO HISTORY BUFFER
            track["history"].append((frame_num, top_y, bottom_y, center_y))

            # 2. CLASSIFY DIRECTION
            if track["direction"] == Direction.UNKNOWN:
                track["direction"] = self._classify_direction(track)
                
                # The exact moment direction is found:
                if track["direction"] != Direction.UNKNOWN:
                    print(f"  Track #{track_id} classified as {track['direction'].value}")
                    track["last_edge_y"] = top_y if track["direction"] == Direction.UP else bottom_y
                    
                    # 3. RUN RETROACTIVE CHECK ON THE BUFFER
                    retro_violation = self._check_retroactive_violation(track, signal)
                    if retro_violation and not track["violated"]:
                        track["violated"] = True
                        track["violation_frame"] = retro_violation["frame"]
                        
                        v = {
                            "track_id":       track_id,
                            "frame":          retro_violation["frame"],
                            "timestamp":      datetime.now().isoformat(),
                            "signal_state":   signal.value,
                            "direction":      track["direction"].value,
                            "stop_line_y":    self.stop_line_y,
                            retro_violation["edge_label"]: round(retro_violation["edge_val"], 1),
                            "frames_tracked": track["frames_tracked"],
                        }
                        self.violations.append(v)
                        events.append(
                            f"Vehicle #{track_id} [{track['direction'].value}]: "
                            f"RETROACTIVE RED LIGHT VIOLATION caught at frame {retro_violation['frame']}"
                        )
                    continue # Skip standard check this frame, already processed buffer

            direction = track["direction"]

            # Skip standard checks until direction is known
            if direction == Direction.UNKNOWN:
                continue

            # 4. STANDARD REAL-TIME CHECK (For cars classified BEFORE hitting the line)
            if signal != SignalState.RED:
                track["last_edge_y"] = top_y if direction == Direction.UP else bottom_y
                continue

            # Cooldown check
            if track["violated"]:
                if frame_num - track["violation_frame"] <= VIOLATION_COOLDOWN_FRAMES:
                    track["last_edge_y"] = top_y if direction == Direction.UP else bottom_y
                    continue
                else:
                    track["violated"] = False

            # Minimum track frames
            if track["frames_tracked"] < MIN_TRACK_FRAMES:
                track["last_edge_y"] = top_y if direction == Direction.UP else bottom_y
                continue

            last_edge = track["last_edge_y"]
            if last_edge is None:
                track["last_edge_y"] = top_y if direction == Direction.UP else bottom_y
                continue

            violated = False

            if direction == Direction.UP:
                if top_y <= self.stop_line_y and last_edge > self.stop_line_y:
                    violated      = True
                    edge_val      = top_y
                    edge_label    = "top_y"
                track["last_edge_y"] = top_y

            elif direction == Direction.DOWN:
                if bottom_y >= self.stop_line_y and last_edge < self.stop_line_y:
                    violated      = True
                    edge_val      = bottom_y
                    edge_label    = "bottom_y"
                track["last_edge_y"] = bottom_y

            if violated:
                track["violated"]        = True
                track["violation_frame"] = frame_num

                v = {
                    "track_id":       track_id,
                    "frame":          frame_num,
                    "timestamp":      datetime.now().isoformat(),
                    "signal_state":   signal.value,
                    "direction":      direction.value,
                    "stop_line_y":    self.stop_line_y,
                    edge_label:       round(edge_val, 1),
                    "frames_tracked": track["frames_tracked"],
                }
                self.violations.append(v)
                events.append(
                    f"Vehicle #{track_id} [{direction.value}]: "
                    f"RED LIGHT VIOLATION at frame {frame_num}"
                )

        self._cleanup(frame_num, active_ids)
        return events

    def _cleanup(self, frame_num: int, active_ids: set):
        stale = [tid for tid, t in self.tracks.items()
                 if tid not in active_ids
                 and frame_num - t["last_seen_frame"] > STALE_TRACK_FRAMES]
        for tid in stale:
            del self.tracks[tid]

    def get_direction(self, track_id: int) -> Direction:
        t = self.tracks.get(int(track_id))
        return t["direction"] if t else Direction.UNKNOWN

    def is_violated(self, track_id: int) -> bool:
        t = self.tracks.get(int(track_id))
        return t is not None and t["violated"]

    def get_summary(self) -> dict:
        up_count   = sum(1 for v in self.violations if v["direction"] == "UP")
        down_count = sum(1 for v in self.violations if v["direction"] == "DOWN")
        return {
            "total_violations": len(self.violations),
            "up_violations":    up_count,
            "down_violations":  down_count,
            "violations":       self.violations,
        }


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


# ── Drawing ───────────────────────────────────────────────────────────────────

DIR_COLORS = {
    Direction.UP:      (255, 200, 0),    # cyan-ish
    Direction.DOWN:    (0, 165, 255),    # orange
    Direction.UNKNOWN: (160, 160, 160),  # grey
}
DIR_ARROWS = {
    Direction.UP:      "↑",
    Direction.DOWN:    "↓",
    Direction.UNKNOWN: "?",
}


def draw_stop_line(frame, orig_w, signal: SignalState, stop_line_y: int):
    color     = (0, 0, 255) if signal == SignalState.RED else (200, 200, 200)
    thickness = 4           if signal == SignalState.RED else 2
    cv2.line(frame, (0, stop_line_y), (orig_w, stop_line_y), color, thickness)
    cv2.putText(frame, f"STOP LINE (Y={stop_line_y})",
                (10, stop_line_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_traffic_light(frame, orig_w, light: TrafficLight):
    lx, ly = orig_w - 110, 30
    cv2.rectangle(frame, (lx - 15, ly - 10), (lx + 75, ly + 110), (40, 40, 40), -1)
    cv2.rectangle(frame, (lx - 15, ly - 10), (lx + 75, ly + 110), (180, 180, 180), 2)
    for idx, st in enumerate([SignalState.RED, SignalState.YELLOW, SignalState.GREEN]):
        cy    = ly + 15 + idx * 32
        color = {SignalState.RED: (0,0,255), SignalState.YELLOW: (0,255,255),
                 SignalState.GREEN: (0,255,0)}[st]
        if st == light.state:
            cv2.circle(frame, (lx + 30, cy), 13, color, -1)
            cv2.circle(frame, (lx + 30, cy), 13, (255,255,255), 2)
        else:
            cv2.circle(frame, (lx + 30, cy), 13, (50,50,50), -1)
    cv2.putText(frame, light.state.value, (lx - 10, ly + 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, light.get_bgr(), 2)
    if USE_KEYBOARD:
        cv2.putText(frame, "R/Y/G keys", (lx - 10, ly + 136),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


def build_labels(tracked, detector: BidirectionalRedLightDetector, model) -> list:
    labels = []
    if tracked.tracker_id is None:
        return labels
    for class_id, track_id in zip(tracked.class_id, tracked.tracker_id):
        name      = model.names[class_id]
        direction = detector.get_direction(track_id)
        arrow     = DIR_ARROWS[direction]
        if detector.is_violated(track_id):
            labels.append(f"#{track_id} {name} {arrow} [!RED LIGHT!]")
        else:
            labels.append(f"#{track_id} {name} {arrow}")
    return labels


def draw_direction_indicators(frame, tracked, detector: BidirectionalRedLightDetector):
    """Draw a small colored arrow on each vehicle box indicating detected direction."""
    if tracked.tracker_id is None:
        return
    for i, track_id in enumerate(tracked.tracker_id):
        direction = detector.get_direction(track_id)
        color     = DIR_COLORS[direction]
        x1, y1, x2, y2 = tracked.xyxy[i]
        cx = int((x1 + x2) / 2)
        # Draw arrow above box
        if direction == Direction.UP:
            cv2.arrowedLine(frame, (cx, int(y1) + 20), (cx, int(y1) - 5),
                            color, 2, tipLength=0.4)
        elif direction == Direction.DOWN:
            cv2.arrowedLine(frame, (cx, int(y1) - 5), (cx, int(y1) + 20),
                            color, 2, tipLength=0.4)


def resize_for_display(frame, target_width=1280):
    h, w  = frame.shape[:2]
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)),
                      interpolation=cv2.INTER_AREA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TVS-8: Red Light Detection — Bidirectional Traffic")
    print("=" * 60)
    print(f"Stop line Y   : {STOP_LINE_Y}")
    print(f"Direction det : first {DIRECTION_FRAMES} frames, min move {DIRECTION_MIN_MOVE}px")
    print(f"UP  vehicles  : top edge (y1) crossing upward")
    print(f"DOWN vehicles : bottom edge (y2) crossing downward")
    print(f"Signal mode   : {'KEYBOARD (R/Y/G)' if USE_KEYBOARD else 'AUTO CYCLE'}")
    print("-" * 60)

    model    = YOLO(MODEL_PATH)
    tracker  = OcclusionTracker(frame_rate=30)
    light    = TrafficLight()
    detector = BidirectionalRedLightDetector(STOP_LINE_Y)

    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)

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
        results    = model(frame,
                           classes=VEHICLE_CLASSES,
                           conf=CONFIDENCE,
                           iou=IOU,
                           verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        tracked, _ = tracker.update(detections, frame_count)

        # Violation check
        events = detector.process(tracked, light.state, frame_count)
        for ev in events:
            print(f"  Frame {frame_count}: {ev}")

        # Draw scene
        annotated = frame.copy()
        draw_stop_line(annotated, orig_w, light.state, STOP_LINE_Y)
        draw_traffic_light(annotated, orig_w, light)
        draw_direction_indicators(annotated, tracked, detector)

        labels    = build_labels(tracked, detector, model)
        annotated = box_annotator.annotate(scene=annotated, detections=tracked)
        annotated = label_annotator.annotate(scene=annotated,
                                              detections=tracked,
                                              labels=labels)

        # Legend
        cv2.putText(annotated, "Arrows: cyan=UP  orange=DOWN  grey=unknown",
                    (10, orig_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # HUD
        active = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
        cv2.putText(annotated,
                    f"Frame: {frame_count} | Active: {active} "
                    f"| Violations: {len(detector.violations)} "
                    f"| Signal: {light.state.value}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        display = resize_for_display(annotated, DISPLAY_WIDTH)
        cv2.imshow("TVS-8 Red Light — Bidirectional", display)

    cap.release()
    cv2.destroyAllWindows()

    # Final report
    summary = detector.get_summary()
    print("\n" + "=" * 60)
    print("RED LIGHT DETECTION REPORT")
    print("=" * 60)
    print(f"Total violations  : {summary['total_violations']}")
    print(f"  UP  direction   : {summary['up_violations']}")
    print(f"  DOWN direction  : {summary['down_violations']}")
    if summary["violations"]:
        print("\nViolation log:")
        for v in summary["violations"]:
            print(f"  Frame {v['frame']:>5}: Vehicle #{v['track_id']:>3}  "
                  f"dir={v['direction']:<4}  "
                  f"tracked={v['frames_tracked']} frames  "
                  f"{v['timestamp']}")

    out = "red_light_violations.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
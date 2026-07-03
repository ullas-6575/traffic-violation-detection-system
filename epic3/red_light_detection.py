"""
red_light_detection.py
TVS-8: Red light crossing detection.

Vehicles move UPWARD (Y decreases) — same camera as TVS-7.
Uses TOP EDGE (y1) of bounding box for stop line crossing detection.

Fixes applied:
  1. Key is read ONCE per loop and passed to signal update correctly
  2. Crossing check uses top edge + upward direction (y1 decreasing)
  3. last_seen is properly tracked so stale cleanup works
  4. Signal state persists correctly between frames
  5. Manual signal change reflected immediately on screen
"""

import cv2
import supervision as sv
from ultralytics import YOLO
import math
import json
from datetime import datetime
from enum import Enum

# ========== CONFIGURATION ==========
VIDEO_PATH      = "videos/tr.mp4"
MODEL_PATH      = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE      = 0.5
IOU             = 0.45
DISPLAY_WIDTH   = 1280

# --- STOP LINE ---
# Vehicles move UP (Y decreases). They must NOT cross this line on RED.
# Place this line where vehicles should stop.
# Vehicles cross it when their TOP EDGE goes from > STOP_LINE_Y to <= STOP_LINE_Y
STOP_LINE_Y = 400

# --- SIGNAL CONTROL ---
# USE_KEYBOARD = True  → press R / Y / G to change signal manually
# USE_KEYBOARD = False → automatic timed cycle
USE_KEYBOARD    = True
LIGHT_CYCLE_FRAMES = 300
GREEN_FRAMES    = 120
YELLOW_FRAMES   = 60
RED_FRAMES      = 120

# --- VIOLATION SETTINGS ---
VIOLATION_COOLDOWN_FRAMES = 90   # ignore re-trigger of same vehicle for N frames
MIN_TRACK_FRAMES          = 5    # vehicle must be tracked this long before triggering
STALE_TRACK_FRAMES        = 100  # remove tracks not seen for this many frames
# ==================================


class SignalState(Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    RED    = "RED"


class TrafficLight:
    """Traffic light signal — keyboard or automatic cycle."""

    def __init__(self):
        self.state = SignalState.GREEN   # current state, persists between frames

    def set_state(self, key: int):
        """Call with the raw cv2 key value to change state via keyboard."""
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
        """Call each frame when using automatic cycle mode."""
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


class RedLightDetector:
    """
    Detects vehicles crossing the stop line while signal is RED.
    Vehicles move UPWARD — uses TOP EDGE (y1) of bounding box.
    Crossing detected when: top_y goes from > stop_line_y  to  <= stop_line_y
    """

    def __init__(self, stop_line_y: int):
        self.stop_line_y = stop_line_y
        self.tracks      = {}   # track_id → state dict
        self.violations  = []

    def _init_track(self, track_id, top_y, frame_num):
        self.tracks[track_id] = {
            "frames_tracked":   0,
            "last_top_y":       top_y,
            "last_seen_frame":  frame_num,
            "violated":         False,
            "violation_frame":  None,
        }

    def process(self, detections: sv.Detections, signal: SignalState, frame_num: int):
        """
        Process one frame of detections.
        Returns list of violation event strings.
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
            top_y = float(y1)   # TOP EDGE — used for crossing detection

            # Init new track
            if track_id not in self.tracks:
                self._init_track(track_id, top_y, frame_num)

            track = self.tracks[track_id]
            track["frames_tracked"]  += 1
            track["last_seen_frame"]  = frame_num

            # Only check violations during RED
            if signal != SignalState.RED:
                track["last_top_y"] = top_y
                continue

            # Cooldown: skip if recently violated
            if track["violated"]:
                if frame_num - track["violation_frame"] <= VIOLATION_COOLDOWN_FRAMES:
                    track["last_top_y"] = top_y
                    continue
                else:
                    track["violated"] = False   # cooldown expired, allow re-trigger

            # Need minimum tracking frames for confidence
            if track["frames_tracked"] < MIN_TRACK_FRAMES:
                track["last_top_y"] = top_y
                continue

            # Crossing check: top edge crosses stop line going UPWARD (Y decreasing)
            # Was below (>) stop line last frame, now at or above (<=) stop line
            if top_y <= self.stop_line_y and track["last_top_y"] > self.stop_line_y:
                track["violated"]        = True
                track["violation_frame"] = frame_num

                v = {
                    "track_id":       track_id,
                    "frame":          frame_num,
                    "timestamp":      datetime.now().isoformat(),
                    "signal_state":   signal.value,
                    "stop_line_y":    self.stop_line_y,
                    "vehicle_top_y":  round(top_y, 1),
                    "frames_tracked": track["frames_tracked"],
                }
                self.violations.append(v)
                events.append(f"Vehicle #{track_id}: RED LIGHT VIOLATION at frame {frame_num}")

            track["last_top_y"] = top_y

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

    def get_summary(self) -> dict:
        return {
            "total_violations": len(self.violations),
            "violations":       self.violations,
        }


class OcclusionTracker:
    """ByteTrack wrapper with consistent ID reassignment across occlusions."""

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


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_stop_line(frame, orig_w, signal: SignalState, stop_line_y: int):
    color     = (0, 0, 255) if signal == SignalState.RED else (200, 200, 200)
    thickness = 4           if signal == SignalState.RED else 2
    cv2.line(frame, (0, stop_line_y), (orig_w, stop_line_y), color, thickness)
    cv2.putText(frame, f"STOP LINE (Y={stop_line_y})",
                (10, stop_line_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_traffic_light(frame, orig_w, light: TrafficLight):
    lx = orig_w - 110
    ly = 30

    # Housing
    cv2.rectangle(frame, (lx - 15, ly - 10),
                  (lx + 75, ly + 105), (40, 40, 40), -1)
    cv2.rectangle(frame, (lx - 15, ly - 10),
                  (lx + 75, ly + 105), (180, 180, 180), 2)

    bulb_states = [SignalState.RED, SignalState.YELLOW, SignalState.GREEN]
    for idx, st in enumerate(bulb_states):
        cy    = ly + 15 + idx * 32
        color = {
            SignalState.RED:    (0, 0, 255),
            SignalState.YELLOW: (0, 255, 255),
            SignalState.GREEN:  (0, 255, 0),
        }[st]
        if st == light.state:
            cv2.circle(frame, (lx + 30, cy), 13, color, -1)
            cv2.circle(frame, (lx + 30, cy), 13, (255, 255, 255), 2)
        else:
            cv2.circle(frame, (lx + 30, cy), 13, (50, 50, 50), -1)

    cv2.putText(frame, light.state.value,
                (lx - 10, ly + 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, light.get_bgr(), 2)
    if USE_KEYBOARD:
        cv2.putText(frame, "R/Y/G keys",
                    (lx - 10, ly + 136),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


def build_labels(tracked, detector: RedLightDetector, model) -> list:
    labels = []
    if tracked.tracker_id is None:
        return labels
    for class_id, track_id in zip(tracked.class_id, tracked.tracker_id):
        name = model.names[class_id]
        if detector.is_violated(track_id):
            labels.append(f"#{track_id} {name} [!RED LIGHT!]")
        else:
            labels.append(f"#{track_id} {name}")
    return labels


def resize_for_display(frame, target_width=1280):
    h, w  = frame.shape[:2]
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)),
                      interpolation=cv2.INTER_AREA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TVS-8: Red Light Crossing Detection")
    print("=" * 60)
    print(f"Direction    : vehicles move UP (Y decreases)")
    print(f"Edge used    : TOP edge (y1) of bounding box")
    print(f"Stop line Y  : {STOP_LINE_Y}")
    print(f"Signal mode  : {'KEYBOARD (R/Y/G)' if USE_KEYBOARD else 'AUTO CYCLE'}")
    print(f"Cooldown     : {VIOLATION_COOLDOWN_FRAMES} frames")
    print(f"Min track    : {MIN_TRACK_FRAMES} frames")
    print("-" * 60)

    model    = YOLO(MODEL_PATH)
    tracker  = OcclusionTracker(frame_rate=30)
    light    = TrafficLight()
    detector = RedLightDetector(STOP_LINE_Y)

    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open video:", VIDEO_PATH)
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video        : {orig_w}x{orig_h}")
    print("Controls     : Q=quit  P=pause  R=Red  Y=Yellow  G=Green")
    print("=" * 60)

    frame_count = 0
    paused      = False

    while True:
        # ── Read key FIRST so it's available for signal update this frame ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print("  [PAUSED]" if paused else "  [RESUMED]")

        # Signal update — keyboard or auto — happens every iteration
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

        # Violation check — uses current light.state (already updated above)
        events = detector.process(tracked, light.state, frame_count)
        for ev in events:
            print(f"  Frame {frame_count}: {ev}")

        # Draw
        annotated = frame.copy()
        draw_stop_line(annotated, orig_w, light.state, STOP_LINE_Y)
        draw_traffic_light(annotated, orig_w, light)

        labels    = build_labels(tracked, detector, model)
        annotated = box_annotator.annotate(scene=annotated, detections=tracked)
        annotated = label_annotator.annotate(scene=annotated,
                                              detections=tracked,
                                              labels=labels)

        # HUD
        active = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
        cv2.putText(annotated,
                    f"Frame: {frame_count} | Active: {active} "
                    f"| Violations: {len(detector.violations)} "
                    f"| Signal: {light.state.value}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        display = resize_for_display(annotated, DISPLAY_WIDTH)
        cv2.imshow("TVS-8 Red Light Detection", display)

    cap.release()
    cv2.destroyAllWindows()

    # Report
    summary = detector.get_summary()
    print("\n" + "=" * 60)
    print("RED LIGHT DETECTION REPORT")
    print("=" * 60)
    print(f"Total violations: {summary['total_violations']}")
    if summary["violations"]:
        print("\nViolation log:")
        for v in summary["violations"]:
            print(f"  Frame {v['frame']:>5}: Vehicle #{v['track_id']:>3}  "
                  f"top_y={v['vehicle_top_y']}  "
                  f"tracked={v['frames_tracked']} frames  "
                  f"time={v['timestamp']}")

    out = "red_light_violations.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
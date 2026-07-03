"""
speed_estimation_simple_fixed.py
TVS-7: Simple, robust speed estimation without complex ID transfer.

Key insight: Instead of trying to fix ByteTrack ID changes,
we track vehicles by their recent position history.
If a new detection is near a recently lost track, we continue its state.
"""

import cv2
import supervision as sv
from ultralytics import YOLO
import math
import json

# ========== CONFIGURATION ==========
VIDEO_PATH = "videos/up.mp4"
MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE = 0.5
IOU = 0.45
DISPLAY_WIDTH = 1280

# Speed lines - vehicles move UP (Y decreases)
START_LINE_Y = 600   # Lower line - crossed FIRST
END_LINE_Y = 300    # Upper line - crossed SECOND

REAL_DISTANCE_METERS = 3.0
SPEED_LIMIT_KMH = 60.0
VIDEO_FPS = 60.0
MIN_FRAMES_VALID = 3

# Ghost track settings
GHOST_FRAMES = 10      # How long to remember lost tracks
REASSIGN_DIST = 100    # Max distance for considering same vehicle
# ==================================


class SimpleSpeedEstimator:
    """
    Simple speed estimator that handles occlusion naturally.
    Uses position-based ghost tracking instead of ByteTrack ID persistence.
    """
    
    def __init__(self, start_y, end_y, real_distance_m, fps, speed_limit):
        self.start_y = start_y
        self.end_y = end_y
        self.real_distance_m = real_distance_m
        self.fps = fps
        self.speed_limit = speed_limit
        self.min_frames = MIN_FRAMES_VALID
        
        # Active tracks: {track_id: state}
        self.active_tracks = {}
        # Ghost tracks: recently lost tracks we might match
        self.ghost_tracks = []
        # Completed measurements
        self.completed = []
        self.discarded = 0
        self.next_id = 1
        
    def _find_matching_ghost(self, cx, cy):
        """Find a ghost track near this position."""
        best_match = None
        best_dist = float('inf')
        
        for ghost in self.ghost_tracks:
            if ghost["frames_since_lost"] > GHOST_FRAMES:
                continue
            gx, gy = ghost["last_pos"]
            dist = math.sqrt((cx - gx)**2 + (cy - gy)**2)
            if dist < REASSIGN_DIST and dist < best_dist:
                best_dist = dist
                best_match = ghost
        
        return best_match
    
    def update(self, detections, frame_num):
        """
        Process all detections in a frame.
        detections: sv.Detections object
        """
        if detections.tracker_id is None:
            # All active tracks are lost
            for tid, state in list(self.active_tracks.items()):
                self._ghost_track(tid, state, frame_num)
            self.active_tracks.clear()
            return
        
        current_ids = set(detections.tracker_id.tolist())
        lost_ids = set(self.active_tracks.keys()) - current_ids
        
        # Mark lost tracks as ghosts
        for tid in lost_ids:
            self._ghost_track(tid, self.active_tracks[tid], frame_num)
            del self.active_tracks[tid]
        
        # Process current detections
        for i, track_id in enumerate(detections.tracker_id.tolist()):
            x1, y1, x2, y2 = detections.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            top_y = y1
            bottom_y = y2
            
            # Check if this is a reassigned ID (near a ghost)
            ghost = self._find_matching_ghost(cx, cy)
            
            if track_id not in self.active_tracks:
                # New track - either fresh or reassigned from ghost
                if ghost is not None:
                    # Continue from ghost state
                    self.active_tracks[track_id] = ghost["state"].copy()
                    self.active_tracks[track_id]["is_ghost_continued"] = True
                    # Remove ghost so it won't match again
                    self.ghost_tracks.remove(ghost)
                else:
                    # Fresh new track
                    self.active_tracks[track_id] = {
                        "state": "idle",
                        "start_frame": None,
                        "end_frame": None,
                        "last_top_y": None,
                        "frames_active": 0,
                        "is_ghost_continued": False
                    }
            
            # Update this track
            state = self.active_tracks[track_id]
            state["frames_active"] += 1
            event, data = self._process_track(state, top_y, bottom_y, frame_num, track_id)
            
            # Update position for next frame
            state["last_pos"] = (cx, cy)
            state["last_top_y"] = top_y
        
        # Clean old ghosts
        self.ghost_tracks = [g for g in self.ghost_tracks 
                            if g["frames_since_lost"] <= GHOST_FRAMES]
        for g in self.ghost_tracks:
            g["frames_since_lost"] += 1
    
    def _ghost_track(self, track_id, state, frame_num):
        """Save a lost track as ghost for potential reassignment."""
        if "last_pos" in state:
            self.ghost_tracks.append({
                "last_pos": state["last_pos"],
                "state": state.copy(),
                "frames_since_lost": 0,
                "lost_frame": frame_num
            })
    
    def _process_track(self, state, top_y, bottom_y, frame_num, track_id):
        """Process a single track's state machine."""
        
        if state["state"] in ("done", "discarded"):
            return None, None
        
        # IDLE
        if state["state"] == "idle":
            state["last_top_y"] = top_y
            
            # Already past both lines
            if top_y < self.end_y:
                state["state"] = "discarded"
                self.discarded += 1
                return "MISSED_BOTH", None
            
            # Between lines - might have missed start
            if top_y < self.start_y:
                # Be lenient: if this is a ghost continuation, it was already timing
                if state.get("is_ghost_continued"):
                    # Was already timing before occlusion
                    return None, None
                # Otherwise, try to recover by pretending it was below
                state["state"] = "waiting_start"
                state["last_top_y"] = self.start_y + 20
                return None, None
            
            state["state"] = "waiting_start"
            return None, None
        
        # WAITING_START
        if state["state"] == "waiting_start":
            if top_y <= self.start_y and state["last_top_y"] > self.start_y:
                state["state"] = "timing"
                state["start_frame"] = frame_num
                return "STARTED_TIMING", frame_num
            
            state["last_top_y"] = top_y
            return None, None
        
        # TIMING
        if state["state"] == "timing":
            if top_y <= self.end_y and state["last_top_y"] > self.end_y:
                state["end_frame"] = frame_num
                frames_between = state["end_frame"] - state["start_frame"]
                
                if frames_between >= self.min_frames:
                    time_s = frames_between / self.fps
                    speed_kmh = round((self.real_distance_m / time_s) * 3.6, 1)
                    
                    measurement = {
                        "track_id": track_id,
                        "start_frame": state["start_frame"],
                        "end_frame": state["end_frame"],
                        "frames_between": frames_between,
                        "time_seconds": round(time_s, 3),
                        "speed_kmh": speed_kmh,
                        "violation": speed_kmh > self.speed_limit,
                    }
                    self.completed.append(measurement)
                    state["state"] = "done"
                    return "SPEED_MEASURED", speed_kmh
                else:
                    state["state"] = "discarded"
                    self.discarded += 1
                    return "TOO_FAST_DISCARDED", None
            
            state["last_top_y"] = top_y
            return None, None
        
        return None, None
    
    def get_speed(self, track_id):
        for m in self.completed:
            if m["track_id"] == track_id:
                return m["speed_kmh"]
        return None
    
    def is_violation(self, track_id):
        for m in self.completed:
            if m["track_id"] == track_id:
                return m["violation"]
        return False
    
    def get_track_state(self, track_id):
        if track_id in self.active_tracks:
            return self.active_tracks[track_id]["state"]
        return "unknown"
    
    def get_summary(self):
        total = len(self.completed)
        violations = sum(1 for m in self.completed if m["violation"])
        avg_speed = sum(m["speed_kmh"] for m in self.completed) / total if total else 0
        return {
            "total_valid": total,
            "discarded": self.discarded,
            "violations": violations,
            "average_speed": round(avg_speed, 1),
            "speed_limit": self.speed_limit,
            "real_distance_m": self.real_distance_m,
            "measurements": self.completed,
        }


def resize_for_display(frame, target_width=1280):
    h, w = frame.shape[:2]
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_overlays(frame, orig_w):
    annotated = frame.copy()
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, END_LINE_Y), (orig_w, START_LINE_Y), (255, 255, 0), -1)
    annotated = cv2.addWeighted(annotated, 0.85, overlay, 0.15, 0)
    
    cv2.line(annotated, (0, START_LINE_Y), (orig_w, START_LINE_Y), (0, 255, 0), 3)
    cv2.putText(annotated, f"START LINE (Y={START_LINE_Y})", (10, START_LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.line(annotated, (0, END_LINE_Y), (orig_w, END_LINE_Y), (0, 0, 255), 3)
    cv2.putText(annotated, f"END LINE (Y={END_LINE_Y})", (10, END_LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return annotated


def build_labels(tracked, speed_est, model):
    labels = []
    if tracked.tracker_id is None:
        return labels
    for class_id, track_id in zip(tracked.class_id, tracked.tracker_id):
        class_name = model.names[class_id]
        speed = speed_est.get_speed(track_id)
        st = speed_est.get_track_state(track_id)
        
        if speed is not None:
            label = f"#{track_id} {class_name} {speed}km/h"
            if speed_est.is_violation(track_id):
                label += " [!VIO]"
        elif st == "timing":
            label = f"#{track_id} {class_name} [timing]"
        elif st == "waiting_start":
            label = f"#{track_id} {class_name} [wait]"
        elif st == "discarded":
            label = f"#{track_id} {class_name} [disc]"
        else:
            label = f"#{track_id} {class_name}"
        
        labels.append(label)
    return labels


def main():
    print("=" * 60)
    print("TVS-7: Speed Estimation — Simple & Robust")
    print("=" * 60)
    print(f"START LINE Y  : {START_LINE_Y}")
    print(f"END LINE Y    : {END_LINE_Y}")
    print(f"Real distance : {REAL_DISTANCE_METERS} m")
    print(f"Speed limit   : {SPEED_LIMIT_KMH} km/h")
    print("-" * 60)
    print("Ghost frames  :", GHOST_FRAMES)
    print("Reassign dist :", REASSIGN_DIST, "px")
    print("=" * 60)
    
    model = YOLO(MODEL_PATH)
    
    # Simple ByteTrack - no complex wrapper
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=VIDEO_FPS,
    )
    
    speed_est = SimpleSpeedEstimator(
        start_y=START_LINE_Y,
        end_y=END_LINE_Y,
        real_distance_m=REAL_DISTANCE_METERS,
        fps=VIDEO_FPS,
        speed_limit=SPEED_LIMIT_KMH,
    )
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo: {orig_w}x{orig_h}")
    print("Press Q=quit, P=pause")
    print("=" * 60)
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect
            results = model(frame, classes=VEHICLE_CLASSES, conf=CONFIDENCE, iou=IOU, verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Track
            tracked = tracker.update_with_detections(detections)
            
            # Speed estimation
            speed_est.update(tracked, frame_count)
            
            # Print events
            if tracked.tracker_id is not None:
                for i, track_id in enumerate(tracked.tracker_id):
                    # We can't easily get events per-track here, so skip console spam
                    pass
            
            # Draw
            annotated = draw_overlays(frame, orig_w)
            labels = build_labels(tracked, speed_est, model)
            annotated = box_annotator.annotate(scene=annotated, detections=tracked)
            annotated = label_annotator.annotate(scene=annotated,
                                                  detections=tracked,
                                                  labels=labels)
            
            active = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
            hud = (f"Frame: {frame_count} | Active: {active} "
                   f"| Valid: {len(speed_est.completed)} "
                   f"| Discarded: {speed_est.discarded}")
            cv2.putText(annotated, hud, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            display = resize_for_display(annotated, DISPLAY_WIDTH)
            cv2.imshow("TVS-7 Speed Estimation — Simple", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Report
    summary = speed_est.get_summary()
    print("\n" + "=" * 60)
    print("SPEED ESTIMATION REPORT")
    print("=" * 60)
    print(f"Total valid measurements : {summary['total_valid']}")
    print(f"Discarded                : {summary['discarded']}")
    print(f"Violations               : {summary['violations']}")
    print(f"Average speed            : {summary['average_speed']} km/h")
    
    if summary["measurements"]:
        print("\nAll measurements:")
        for m in summary["measurements"]:
            tag = "VIOLATION" if m["violation"] else "OK"
            print(f"  Vehicle #{m['track_id']:>3}: "
                  f"{m['speed_kmh']:>6.1f} km/h  "
                  f"({m['frames_between']} frames)  [{tag}]")
    
    out_path = "speed_measurements_simple.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
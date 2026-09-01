"""
track_lifecycle_fixed.py
TVS-5 Subtask 4: Log track lifecycle events (JSON serialization fixed).

Usage: python track_lifecycle_fixed.py
"""

import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import math
import json
from datetime import datetime

# ========== CONFIG ==========
VIDEO_PATH = "videos/ttt.mp4"
MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE = 0.5
IOU = 0.45

GHOST_BUFFER_FRAMES = 15
REASSIGN_DISTANCE_THRESH = 80
DISPLAY_WIDTH = 1280
LOG_FILE = "track_lifecycle_log.json"
# ============================

def to_python_float(value):
    """Convert numpy float to Python float for JSON serialization."""
    if hasattr(value, 'item'):
        return value.item()
    return float(value)


class OcclusionTrackerWithLogging:
    def __init__(self, frame_rate=30):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=frame_rate
        )
        self.ghost_tracks = {}
        self.id_map = {}
        self.next_consistent_id = 1
        self.consistent_history = defaultdict(list)
        self.events = []
        
    def log_event(self, event_type, track_id, frame_num, details=None):
        event = {
            "timestamp": datetime.now().isoformat(),
            "frame": int(frame_num),
            "event": event_type,
            "track_id": int(track_id),
            "details": details or {}
        }
        self.events.append(event)
        
    def update(self, detections, frame_num):
        prev_active_ids = set(self.id_map.values())
        tracked = self.tracker.update_with_detections(detections)
        
        if tracked.tracker_id is None:
            for tid in prev_active_ids:
                self.log_event("LOST", tid, frame_num, {"reason": "no_detections"})
            return tracked
        
        new_tracker_ids = tracked.tracker_id.tolist()
        current_active_ids = set()
        new_consistent_ids = []
        
        for i, tid in enumerate(new_tracker_ids):
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            if tid in self.id_map:
                consistent_id = self.id_map[tid]
                if frame_num % 10 == 0:
                    self.log_event("UPDATE", consistent_id, frame_num, {
                        "position": [round(to_python_float(cx), 1), round(to_python_float(cy), 1)],
                        "bbox": [
                            round(to_python_float(x1), 1),
                            round(to_python_float(y1), 1),
                            round(to_python_float(x2), 1),
                            round(to_python_float(y2), 1)
                        ]
                    })
            else:
                consistent_id = self._try_reassign(cx, cy, frame_num)
                
                if consistent_id is not None:
                    self.log_event("REASSIGNED", consistent_id, frame_num, {
                        "position": [round(to_python_float(cx), 1), round(to_python_float(cy), 1)],
                        "reason": "occlusion_recovery"
                    })
                else:
                    consistent_id = self.next_consistent_id
                    self.next_consistent_id += 1
                    class_name = model.names[int(tracked.class_id[i])] if tracked.class_id is not None else "unknown"
                    self.log_event("CREATED", consistent_id, frame_num, {
                        "position": [round(to_python_float(cx), 1), round(to_python_float(cy), 1)],
                        "class": class_name
                    })
                
                self.id_map[tid] = consistent_id
            
            current_active_ids.add(consistent_id)
            new_consistent_ids.append(consistent_id)
            self.consistent_history[consistent_id].append((
                to_python_float(cx), to_python_float(cy), int(frame_num)
            ))
        
        lost_ids = prev_active_ids - current_active_ids
        for lid in lost_ids:
            self.log_event("LOST", lid, frame_num, {"reason": "exited_frame_or_occluded"})
        
        self._clean_ghosts(frame_num)
        
        for i, tid in enumerate(new_tracker_ids):
            cid = self.id_map[tid]
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            class_id = int(tracked.class_id[i])
            self.ghost_tracks[cid] = {
                "last_pos": (to_python_float(cx), to_python_float(cy)),
                "lost_frame": int(frame_num),
                "class": class_id
            }
        
        tracked.tracker_id = new_consistent_ids
        return tracked
    
    def _try_reassign(self, cx, cy, frame_num):
        best_match = None
        best_dist = float('inf')
        
        for gid, ghost in self.ghost_tracks.items():
            if frame_num - ghost["lost_frame"] > GHOST_BUFFER_FRAMES:
                continue
            
            gx, gy = ghost["last_pos"]
            dist = math.sqrt((to_python_float(cx) - gx)**2 + (to_python_float(cy) - gy)**2)
            
            if dist < REASSIGN_DISTANCE_THRESH and dist < best_dist:
                best_dist = dist
                best_match = gid
        
        if best_match is not None:
            del self.ghost_tracks[best_match]
            return best_match
        
        return None
    
    def _clean_ghosts(self, frame_num):
        to_remove = []
        for gid, ghost in self.ghost_tracks.items():
            if frame_num - ghost["lost_frame"] > GHOST_BUFFER_FRAMES:
                to_remove.append(gid)
        for gid in to_remove:
            del self.ghost_tracks[gid]
    
    def save_log(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.events, f, indent=2)
        print(f"\nLog saved: {filename} ({len(self.events)} events)")


def resize_for_display(frame, target_width=1280):
    h, w = frame.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def main():
    print("=" * 60)
    print("Track Lifecycle Logging (Fixed)")
    print("=" * 60)
    
    global model
    model = YOLO(MODEL_PATH)
    occ_tracker = OcclusionTrackerWithLogging(frame_rate=30)
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {orig_w}x{orig_h} @ {fps:.1f} FPS | {total} frames")
    print(f"Logging to: {LOG_FILE}")
    print("-" * 60)
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
            
            results = model(frame, classes=VEHICLE_CLASSES, conf=CONFIDENCE, iou=IOU, verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])
            tracked = occ_tracker.update(detections, frame_count)
            
            labels = []
            if tracked.tracker_id is not None:
                for class_id, track_id in zip(tracked.class_id, tracked.tracker_id):
                    class_name = model.names[class_id]
                    labels.append(f"#{track_id} {class_name}")
            
            annotated = frame.copy()
            
            for gid, ghost in occ_tracker.ghost_tracks.items():
                px, py = int(ghost["last_pos"][0]), int(ghost["last_pos"][1])
                age = frame_count - ghost["lost_frame"]
                alpha = max(0, 1 - age / GHOST_BUFFER_FRAMES)
                color = (0, int(255 * alpha), int(255 * alpha))
                cv2.circle(annotated, (px, py), 8, color, 2)
            
            annotated = box_annotator.annotate(scene=annotated, detections=tracked)
            annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)
            
            active = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
            ghosts = len(occ_tracker.ghost_tracks)
            total_unique = occ_tracker.next_consistent_id - 1
            
            info = f"Frame: {frame_count} | Active: {active} | Ghosts: {ghosts} | Total IDs: {total_unique}"
            cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            display_frame = resize_for_display(annotated, DISPLAY_WIDTH)
            cv2.imshow("Lifecycle Logging", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    
    occ_tracker.save_log(LOG_FILE)
    
    print("\n" + "=" * 60)
    print("LIFECYCLE LOGGING REPORT")
    print("=" * 60)
    print(f"Frames: {frame_count}")
    print(f"Total consistent IDs: {occ_tracker.next_consistent_id - 1}")
    
    event_counts = defaultdict(int)
    for e in occ_tracker.events:
        event_counts[e["event"]] += 1
    
    print(f"\nEvent counts:")
    for event_type, count in sorted(event_counts.items()):
        print(f"  {event_type}: {count}")
    
    print("\nFirst 3 events:")
    for e in occ_tracker.events[:3]:
        print(f"  Frame {e['frame']}: {e['event']} track #{e['track_id']}")
    
    print("\nLast 3 events:")
    for e in occ_tracker.events[-3:]:
        print(f"  Frame {e['frame']}: {e['event']} track #{e['track_id']}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
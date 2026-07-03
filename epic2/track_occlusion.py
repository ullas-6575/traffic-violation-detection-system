"""
track_occlusion.py (resized display version)
TVS-5 Subtask 3: Handle ID reassignment on occlusion.

Usage: python track_occlusion_resized.py
"""

import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import math

# ========== CONFIG ==========
VIDEO_PATH = "videos/ttt.mp4"
MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE = 0.5
IOU = 0.45

# Occlusion handling
GHOST_BUFFER_FRAMES = 15
REASSIGN_DISTANCE_THRESH = 80
MIN_TRACK_LENGTH = 5

# Display resize
DISPLAY_WIDTH = 1280   # resize for display (detection still uses full res)
DISPLAY_HEIGHT = 720   # set to None to keep aspect ratio
# ============================

class OcclusionTracker:
    """Wrapper around ByteTrack that handles ID reassignment on occlusion."""
    
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
        
    def update(self, detections, frame_num):
        tracked = self.tracker.update_with_detections(detections)
        
        if tracked.tracker_id is None:
            return tracked
        
        new_tracker_ids = tracked.tracker_id.tolist()
        new_consistent_ids = []
        
        for i, tid in enumerate(new_tracker_ids):
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            if tid in self.id_map:
                consistent_id = self.id_map[tid]
            else:
                consistent_id = self._try_reassign(cx, cy, frame_num)
                
                if consistent_id is None:
                    consistent_id = self.next_consistent_id
                    self.next_consistent_id += 1
                
                self.id_map[tid] = consistent_id
            
            new_consistent_ids.append(consistent_id)
            self.consistent_history[consistent_id].append((cx, cy, frame_num))
        
        self._clean_ghosts(frame_num)
        
        for i, tid in enumerate(new_tracker_ids):
            cid = self.id_map[tid]
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            class_id = tracked.class_id[i]
            self.ghost_tracks[cid] = {
                "last_pos": (cx, cy),
                "lost_frame": frame_num,
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
            dist = math.sqrt((cx - gx)**2 + (cy - gy)**2)
            
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


def resize_for_display(frame, target_width=1280, target_height=None):
    """Resize frame for display while keeping aspect ratio."""
    h, w = frame.shape[:2]
    
    if target_height is None:
        scale = target_width / w
        new_w = target_width
        new_h = int(h * scale)
    else:
        new_w = target_width
        new_h = target_height
    
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    print("=" * 60)
    print("Occlusion-Aware Vehicle Tracking (Resized Display)")
    print("=" * 60)
    
    model = YOLO(MODEL_PATH)
    occ_tracker = OcclusionTracker(frame_rate=30)
    
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
    
    print(f"Original video: {orig_w}x{orig_h}")
    print(f"Display size: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT or 'auto'}")
    print(f"FPS: {fps:.1f} | Frames: {total}")
    print("-" * 60)
    print("Press Q=quit, P=pause, +/- to adjust display size")
    print("=" * 60)
    
    frame_count = 0
    paused = False
    current_display_width = DISPLAY_WIDTH
    
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
            
            # Annotate on ORIGINAL frame (full resolution)
            annotated = frame.copy()
            
            # Draw ghost tracks
            for gid, ghost in occ_tracker.ghost_tracks.items():
                px, py = int(ghost["last_pos"][0]), int(ghost["last_pos"][1])
                age = frame_count - ghost["lost_frame"]
                alpha = max(0, 1 - age / GHOST_BUFFER_FRAMES)
                color = (0, int(255 * alpha), int(255 * alpha))
                cv2.circle(annotated, (px, py), 8, color, 2)
            
            annotated = box_annotator.annotate(scene=annotated, detections=tracked)
            annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)
            
            # Info on original frame
            active = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
            ghosts = len(occ_tracker.ghost_tracks)
            total_unique = occ_tracker.next_consistent_id - 1
            
            info = f"Frame: {frame_count} | Active: {active} | Ghosts: {ghosts} | Total IDs: {total_unique}"
            cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # RESIZE for display only
            display_frame = resize_for_display(annotated, current_display_width)
            
            cv2.imshow("Occlusion-Aware Tracking", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('+') or key == ord('='):
            current_display_width = min(current_display_width + 100, orig_w)
            print(f"Display width: {current_display_width}")
        elif key == ord('-'):
            current_display_width = max(current_display_width - 100, 640)
            print(f"Display width: {current_display_width}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("OCCLUSION TRACKING REPORT")
    print("=" * 60)
    print(f"Frames: {frame_count}")
    print(f"Total consistent IDs: {occ_tracker.next_consistent_id - 1}")
    print(f"Ghost tracks at end: {len(occ_tracker.ghost_tracks)}")
    
    print("\nLongest tracked vehicles:")
    sorted_hist = sorted(occ_tracker.consistent_history.items(), 
                        key=lambda x: len(x[1]), reverse=True)[:10]
    for cid, hist in sorted_hist:
        duration = len(hist)
        first = hist[0][2]
        last = hist[-1][2]
        print(f"  ID #{cid}: {duration} frames (frames {first}-{last})")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
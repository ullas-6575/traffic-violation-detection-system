"""
overlay_system.py
TVS-6: Complete detection overlay system with virtual lines and production toggle.

Usage:
    python overlay_system.py              # with overlay (development)
    python overlay_system.py --no-overlay # without overlay (production)
"""

import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import math
import argparse

# ========== CONFIG ==========
VIDEO_PATH = "videos/ttt.mp4"
MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE = 0.5
IOU = 0.45
DISPLAY_WIDTH = 1280

# Virtual lines for speed estimation (Epic 3 prep)
# These are Y-coordinates (horizontal lines) - adjust for your video
SPEED_LINE_1_Y = 400   # First speed measurement line
SPEED_LINE_2_Y = 600   # Second speed measurement line

# Stop line for red light detection (Epic 3 prep)
STOP_LINE_Y = 700

# Line colors
SPEED_LINE_COLOR = (0, 255, 255)    # Cyan
STOP_LINE_COLOR = (0, 0, 255)       # Red
LINE_THICKNESS = 2
# ============================

class OcclusionTracker:
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
            self.consistent_history[consistent_id].append((float(cx), float(cy), int(frame_num)))
        
        self._clean_ghosts(frame_num)
        
        for i, tid in enumerate(new_tracker_ids):
            cid = self.id_map[tid]
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            class_id = int(tracked.class_id[i])
            self.ghost_tracks[cid] = {
                "last_pos": (float(cx), float(cy)),
                "lost_frame": int(frame_num),
                "class": class_id
            }
        
        tracked.tracker_id = new_consistent_ids
        return tracked
    
    def _try_reassign(self, cx, cy, frame_num):
        best_match = None
        best_dist = float('inf')
        
        for gid, ghost in self.ghost_tracks.items():
            if frame_num - ghost["lost_frame"] > 15:
                continue
            
            gx, gy = ghost["last_pos"]
            dist = math.sqrt((float(cx) - gx)**2 + (float(cy) - gy)**2)
            
            if dist < 80 and dist < best_dist:
                best_dist = dist
                best_match = gid
        
        if best_match is not None:
            del self.ghost_tracks[best_match]
            return best_match
        
        return None
    
    def _clean_ghosts(self, frame_num):
        to_remove = []
        for gid, ghost in self.ghost_tracks.items():
            if frame_num - ghost["lost_frame"] > 15:
                to_remove.append(gid)
        for gid in to_remove:
            del self.ghost_tracks[gid]


class OverlayRenderer:
    """Handles all visual overlays for the detection system."""
    
    def __init__(self, show_overlay=True):
        self.show_overlay = show_overlay
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)
        
    def draw_virtual_lines(self, frame, frame_width):
        """Draw speed measurement and stop lines."""
        if not self.show_overlay:
            return frame
            
        annotated = frame.copy()
        
        # Speed measurement lines
        cv2.line(annotated, (0, SPEED_LINE_1_Y), (frame_width, SPEED_LINE_1_Y), 
                SPEED_LINE_COLOR, LINE_THICKNESS)
        cv2.line(annotated, (0, SPEED_LINE_2_Y), (frame_width, SPEED_LINE_2_Y), 
                SPEED_LINE_COLOR, LINE_THICKNESS)
        
        # Labels for speed lines
        cv2.putText(annotated, "SPEED LINE 1", (10, SPEED_LINE_1_Y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, SPEED_LINE_COLOR, 1)
        cv2.putText(annotated, "SPEED LINE 2", (10, SPEED_LINE_2_Y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, SPEED_LINE_COLOR, 1)
        
        # Stop line
        cv2.line(annotated, (0, STOP_LINE_Y), (frame_width, STOP_LINE_Y), 
                STOP_LINE_COLOR, LINE_THICKNESS + 1)
        cv2.putText(annotated, "STOP LINE", (10, STOP_LINE_Y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, STOP_LINE_COLOR, 1)
        
        # Legend
        legend_y = 30
        cv2.putText(annotated, "VIRTUAL LINES:", (frame_width - 250, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, "Cyan = Speed measurement", (frame_width - 250, legend_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, SPEED_LINE_COLOR, 1)
        cv2.putText(annotated, "Red = Stop line (red light)", (frame_width - 250, legend_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, STOP_LINE_COLOR, 1)
        
        return annotated
    
    def draw_detections(self, frame, tracked, model_names):
        """Draw bounding boxes, labels, and traces."""
        if not self.show_overlay:
            return frame
            
        labels = []
        if tracked.tracker_id is not None:
            for class_id, track_id in zip(tracked.class_id, tracked.tracker_id):
                class_name = model_names[class_id]
                labels.append(f"ID:{track_id} {class_name}")
        
        annotated = self.box_annotator.annotate(scene=frame, detections=tracked)
        annotated = self.label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)
        
        return annotated
    
    def draw_info_panel(self, frame, frame_num, active_tracks, total_ids, fps=None):
        """Draw information panel on frame."""
        if not self.show_overlay:
            return frame
            
        info_lines = [
            f"Frame: {frame_num}",
            f"Active: {active_tracks}",
            f"Total IDs: {total_ids}",
        ]
        if fps:
            info_lines.append(f"FPS: {fps:.1f}")
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # Mode indicator
        mode_text = "DEV MODE (overlay ON)" if self.show_overlay else "PROD MODE (overlay OFF)"
        mode_color = (0, 255, 0) if self.show_overlay else (0, 165, 255)
        cv2.putText(frame, mode_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        return frame


def resize_for_display(frame, target_width=1280):
    h, w = frame.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection Overlay System")
    parser.add_argument("--no-overlay", action="store_true", 
                       help="Disable visual overlay (production mode)")
    args = parser.parse_args()
    
    show_overlay = not args.no_overlay
    
    print("=" * 60)
    print("Epic 2 Complete: Detection Overlay System")
    print("=" * 60)
    print(f"Overlay mode: {'ON (development)' if show_overlay else 'OFF (production)'}")
    print("-" * 60)
    
    model = YOLO(MODEL_PATH)
    tracker = OcclusionTracker(frame_rate=30)
    renderer = OverlayRenderer(show_overlay=show_overlay)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {orig_w}x{orig_h} @ {fps:.1f} FPS | {total} frames")
    print("-" * 60)
    print("Controls: Q=quit, P=pause, O=toggle overlay")
    print("=" * 60)
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detection
            results = model(frame, classes=VEHICLE_CLASSES, conf=CONFIDENCE, iou=IOU, verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])
            tracked = tracker.update(detections, frame_count)
            
            # Overlay rendering
            if show_overlay:
                # Draw virtual lines first (behind detections)
                frame = renderer.draw_virtual_lines(frame, orig_w)
                
                # Draw detections
                frame = renderer.draw_detections(frame, tracked, model.names)
                
                # Draw info panel
                active = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
                total_ids = tracker.next_consistent_id - 1
                frame = renderer.draw_info_panel(frame, frame_count, active, total_ids)
                
                # Resize for display
                display_frame = resize_for_display(frame, DISPLAY_WIDTH)
                cv2.imshow("Vehicle Detection System", display_frame)
            else:
                # Production mode: no display, just process
                if frame_count % 100 == 0:
                    active = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
                    print(f"Frame {frame_count}: {active} active tracks")
        
        key = cv2.waitKey(1 if show_overlay else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('o') and show_overlay:
            # Toggle overlay on/off in real-time
            show_overlay = not show_overlay
            renderer.show_overlay = show_overlay
            print(f"Overlay: {'ON' if show_overlay else 'OFF'}")
    
    cap.release()
    if show_overlay:
        cv2.destroyAllWindows()
    
    # Final report
    print("\n" + "=" * 60)
    print("EPIC 2 COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Frames processed: {frame_count}")
    print(f"Total unique vehicles: {tracker.next_consistent_id - 1}")
    print(f"Overlay mode: {'Development' if show_overlay else 'Production'}")
    
    print("\nEpic 2 Stories Completed:")
    print("  TVS-4: YOLOv8 inference pipeline")
    print("  TVS-5: Vehicle tracker with occlusion handling")
    print("  TVS-6: Detection overlay with virtual lines")
    
    print("\nReady for Epic 3: Violation Detection Engine")
    print("  - Speed estimation (virtual line timing)")
    print("  - Red light crossing detection")
    print("  - Violation rule engine")
    print("=" * 60)

if __name__ == "__main__":
    main()
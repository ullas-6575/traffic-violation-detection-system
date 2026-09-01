"""
track_persistent.py
TVS-5 Subtask 2: Persistent track IDs with updated supervision API.

Usage: python track_persistent.py
"""

import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

# ========== CONFIG ==========
VIDEO_PATH = "videos/ttt.mp4"  # change to your video
MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE = 0.5
IOU = 0.45
MAX_FRAMES = None  # None = process full video
# ============================

def main():
    print("=" * 60)
    print("Persistent Vehicle Tracking")
    print("=" * 60)
    
    # Load model
    print("\nLoading YOLOv8n...")
    model = YOLO(MODEL_PATH)
    
    # Initialize tracker with new API
    print("Initializing tracker...")
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30
    )
    
    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)
    
    # Track history storage: {track_id: [(x, y, frame_num), ...]}
    track_history = defaultdict(list)
    
    # Track metadata: {track_id: {"first_seen": frame, "class": name, "last_seen": frame}}
    track_metadata = {}
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {w}x{h} @ {fps:.1f} FPS | {total} frames")
    print("-" * 60)
    print("Press Q=quit, P=pause, S=save snapshot")
    print("=" * 60)
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nVideo finished.")
                break
            
            frame_count += 1
            
            if MAX_FRAMES and frame_count > MAX_FRAMES:
                break
            
            # Detect
            results = model(
                frame, 
                classes=VEHICLE_CLASSES, 
                conf=CONFIDENCE, 
                iou=IOU,
                verbose=False
            )
            
            # Convert to supervision detections
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Update tracker
            detections = tracker.update_with_detections(detections)
            
            # Store track history and metadata
            if detections.tracker_id is not None:
                for i, track_id in enumerate(detections.tracker_id):
                    # Get center point of bounding box
                    x1, y1, x2, y2 = detections.xyxy[i]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Store position history
                    track_history[track_id].append((center_x, center_y, frame_count))
                    
                    # Keep only last 100 positions (memory management)
                    if len(track_history[track_id]) > 100:
                        track_history[track_id].pop(0)
                    
                    # Update metadata
                    class_id = detections.class_id[i]
                    class_name = model.names[class_id]
                    
                    if track_id not in track_metadata:
                        track_metadata[track_id] = {
                            "first_seen": frame_count,
                            "class": class_name,
                            "last_seen": frame_count,
                            "total_frames": 0
                        }
                    
                    track_metadata[track_id]["last_seen"] = frame_count
                    track_metadata[track_id]["total_frames"] += 1
            
            # Prepare labels
            labels = []
            if detections.tracker_id is not None:
                for class_id, track_id in zip(detections.class_id, detections.tracker_id):
                    class_name = model.names[class_id]
                    labels.append(f"ID:{track_id} {class_name}")
            
            # Annotate
            annotated = frame.copy()
            
            # Draw traces (paths) for each track
            for track_id, history in track_history.items():
                if len(history) > 1:
                    # Only draw if track is currently active or recently active
                    last_frame = history[-1][2]
                    if frame_count - last_frame <= 5:  # within last 5 frames
                        points = [(p[0], p[1]) for p in history]
                        for i in range(1, len(points)):
                            cv2.line(annotated, points[i-1], points[i], (0, 255, 255), 2)
            
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            
            # Info overlay
            active = len(detections.tracker_id) if detections.tracker_id is not None else 0
            total_unique = len(track_metadata)
            info = f"Frame: {frame_count} | Active: {active} | Total Unique IDs: {total_unique}"
            cv2.putText(annotated, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            annotated = cv2.resize(annotated, (960, 540))  # resize for display
            cv2.imshow("Persistent Tracking", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuit by user.")
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused." if paused else "Resumed.")
        elif key == ord('s'):
            filename = f"snapshot_frame_{frame_count}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"Snapshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final Report
    print("\n" + "=" * 60)
    print("PERSISTENT TRACKING REPORT")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total unique vehicles tracked: {len(track_metadata)}")
    print("-" * 60)
    
    # Sort by first appearance
    sorted_tracks = sorted(track_metadata.items(), key=lambda x: x[1]["first_seen"])
    
    print("\nVehicle Timeline:")
    print(f"{'ID':<6} {'Class':<12} {'First':<8} {'Last':<8} {'Duration':<10}")
    print("-" * 50)
    for track_id, meta in sorted_tracks:
        duration = meta["last_seen"] - meta["first_seen"] + 1
        print(f"{track_id:<6} {meta['class']:<12} {meta['first_seen']:<8} {meta['last_seen']:<8} {duration:<10}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
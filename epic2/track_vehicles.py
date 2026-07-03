"""
track_vehicles.py
TVS-5 Subtask 1: Integrate ByteTrack for persistent vehicle IDs.

Usage: python track_vehicles.py
"""

import cv2
import supervision as sv
from ultralytics import YOLO

# ========== CONFIG ==========
VIDEO_PATH = "videos/t.mp4"
MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CONFIDENCE = 0.5
IOU = 0.45
MAX_FRAMES = 300  # process first 300 frames (set to None for full video)
# ============================

def main():
    print("=" * 60)
    print("Vehicle Tracking - ByteTrack Integration")
    print("=" * 60)
    
    # Load model
    print("\nLoading YOLOv8n...")
    model = YOLO(MODEL_PATH)
    
    # Initialize ByteTrack
    print("Initializing ByteTrack tracker...")
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,  # minimum score to start tracking
        lost_track_buffer=30,              # frames to keep lost tracks
        minimum_matching_threshold=0.8,    # IoU threshold for matching
        frame_rate=30                      # assumed video FPS
    )
    
    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {fps:.1f} | Total frames: {total}")
    print(f"Tracker: ByteTrack")
    print("-" * 60)
    print("Press 'Q' to quit, 'P' to pause")
    print("=" * 60)
    
    frame_count = 0
    paused = False
    active_tracks_history = []  # for stats
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nVideo finished.")
                break
            
            frame_count += 1
            
            if MAX_FRAMES and frame_count > MAX_FRAMES:
                print(f"\nReached max frames ({MAX_FRAMES}).")
                break
            
            # Detect vehicles
            results = model(
                frame, 
                classes=VEHICLE_CLASSES, 
                conf=CONFIDENCE, 
                iou=IOU,
                verbose=False
            )
            
            # Convert to supervision format
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Track detections
            detections = tracker.update_with_detections(detections)
            
            # Prepare labels with track IDs
            labels = []
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
                class_name = model.names[class_id]
                labels.append(f"#{tracker_id} {class_name}")
            
            # Annotate frame
            annotated = frame.copy()
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            
            # Add info overlay
            active_tracks = len(detections.tracker_id) if detections.tracker_id is not None else 0
            active_tracks_history.append(active_tracks)
            
            info = f"Frame: {frame_count} | Active Tracks: {active_tracks}"
            cv2.putText(annotated, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show
            cv2.imshow("Vehicle Tracking - ByteTrack", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuit by user.")
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused." if paused else "Resumed.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "=" * 60)
    print("TRACKING SUMMARY")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    if active_tracks_history:
        avg_active = sum(active_tracks_history) / len(active_tracks_history)
        max_active = max(active_tracks_history)
        print(f"Average active tracks per frame: {avg_active:.1f}")
        print(f"Max simultaneous tracks: {max_active}")
    print("=" * 60)

if __name__ == "__main__":
    main()
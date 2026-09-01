"""
detect_vehicles.py
TVS-4 Subtask 2: Run YOLOv8 on video, filter only vehicle classes.

Usage: python detect_vehicles.py
"""

import cv2
from ultralytics import YOLO

# ========== CONFIG ==========
VIDEO_PATH = "videos/traffic.mp4"  # <-- change to your video filename
MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
CONFIDENCE = 0.5
MAX_FRAMES = 200  # process first 100 frames for quick test (set to None for full video)
# ============================

def main():
    print("=" * 50)
    print("Vehicle Detection - YOLOv8 Filtered")
    print("=" * 50)
    
    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model ready.")
    
    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {VIDEO_PATH}")
        print("Make sure the file exists in the 'videos' folder.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"Filtering classes: {VEHICLE_CLASSES} (car, motorcycle, bus, truck)")
    print(f"Confidence threshold: {CONFIDENCE}")
    print("\nPress 'Q' to quit, 'P' to pause/unpause")
    print("=" * 50)
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nVideo finished.")
                break
            
            frame_count += 1
            
            # Stop early if MAX_FRAMES is set
            if MAX_FRAMES and frame_count > MAX_FRAMES:
                print(f"\nReached max frames limit ({MAX_FRAMES}).")
                break
            
            # Run detection with class filter
            results = model(frame, classes=VEHICLE_CLASSES, conf=CONFIDENCE, verbose=False)
            
            # Draw results on frame
            annotated = results[0].plot()
            
            # Add info text
            detections = len(results[0].boxes)
            info = f"Frame: {frame_count}/{total_frames} | Vehicles: {detections}"
            cv2.putText(annotated, info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            annotated = cv2.resize(annotated, (960, 540))  # resize for display
            # Show frame
            cv2.imshow("Vehicle Detection - YOLOv8", annotated)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuit by user.")
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused." if paused else "Resumed.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessed {frame_count} frames.")
    print("Done!")

if __name__ == "__main__":
    main()
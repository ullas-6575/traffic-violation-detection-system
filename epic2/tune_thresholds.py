"""
tune_thresholds.py
TVS-4 Subtask 4: Test different confidence and NMS thresholds.

Usage: python tune_thresholds.py
Press 'N' to cycle through thresholds, 'Q' to quit
"""

import cv2
from ultralytics import YOLO

# ========== CONFIG ==========
VIDEO_PATH = "videos/t.mp4"
MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]

# Threshold combinations to test
THRESHOLDS = [
    {"name": "Low conf / Low NMS",  "conf": 0.3, "iou": 0.3},
    {"name": "Balanced (default)",  "conf": 0.5, "iou": 0.45},
    {"name": "High conf / Low NMS", "conf": 0.7, "iou": 0.3},
    {"name": "High conf / High NMS", "conf": 0.7, "iou": 0.6},
]
# ============================

def main():
    print("=" * 60)
    print("Threshold Tuning Tool")
    print("=" * 60)
    
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return
    
    # Read first frame for testing
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read video")
        return
    
    current_idx = 1  # start with balanced
    paused = False
    
    print(f"\nLoaded. Testing {len(THRESHOLDS)} threshold combinations.")
    print("Controls:")
    print("  N = next threshold preset")
    print("  P = pause/unpause")
    print("  Q = quit")
    print("-" * 60)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
                continue
        
        thresh = THRESHOLDS[current_idx]
        
        # Run detection with current thresholds
        results = model(
            frame, 
            classes=VEHICLE_CLASSES, 
            conf=thresh["conf"], 
            iou=thresh["iou"],
            verbose=False
        )
        
        # Draw results
        annotated = results[0].plot()
        
        # Add threshold info
        info_lines = [
            f"Preset: {thresh['name']}",
            f"Confidence: {thresh['conf']}  |  NMS IoU: {thresh['iou']}",
            f"Detections: {len(results[0].boxes)}",
            f"Press N to cycle, P to pause, Q to quit"
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(annotated, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        cv2.imshow("Threshold Tuning", annotated)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_idx = (current_idx + 1) % len(THRESHOLDS)
            print(f"Switched to: {THRESHOLDS[current_idx]['name']}")
        elif key == ord('p'):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("-" * 60)
    print("For traffic violation detection, I recommend:")
    print("  conf=0.5, iou=0.45 (Balanced default)")
    print("\nReasons:")
    print("  - 0.5 confidence catches most vehicles without too many false positives")
    print("  - 0.45 NMS removes duplicate boxes while keeping nearby vehicles")
    print("  - Adjust conf to 0.4 if you're missing distant/small vehicles")
    print("  - Adjust conf to 0.6 if you get too many false detections")
    print("=" * 60)

if __name__ == "__main__":
    main()
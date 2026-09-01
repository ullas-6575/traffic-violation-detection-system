"""
benchmark_fps.py
TVS-4 Subtask 3: Benchmark YOLOv8 inference FPS on your CPU.

Usage: python benchmark_fps.py
"""

import cv2
import time
import statistics
from ultralytics import YOLO

# ========== CONFIG ==========
VIDEO_PATH = "videos/tt.mp4"  # <-- your video
MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE = 0.5
BENCHMARK_FRAMES = 200  # frames to benchmark (set higher for more accuracy)
WARMUP_FRAMES = 10      # discard first N frames (cache warmup)
# ============================

def benchmark():
    print("=" * 60)
    print("YOLOv8 FPS Benchmark")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded.")
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return
    
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo: {width}x{height} @ {fps_video:.1f} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Benchmark frames: {BENCHMARK_FRAMES} (warmup: {WARMUP_FRAMES})")
    print(f"Model: YOLOv8n (nano) | Device: CPU")
    print("-" * 60)
    
    inference_times = []
    frame_count = 0
    
    while frame_count < BENCHMARK_FRAMES + WARMUP_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Time only the inference (not reading frame)
        t0 = time.perf_counter()
        results = model(frame, classes=VEHICLE_CLASSES, conf=CONFIDENCE, verbose=False)
        t1 = time.perf_counter()
        
        inference_time = t1 - t0
        
        # Skip warmup frames
        if frame_count >= WARMUP_FRAMES:
            inference_times.append(inference_time)
        
        frame_count += 1
        
        # Progress
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{BENCHMARK_FRAMES + WARMUP_FRAMES} frames...")
    
    cap.release()
    
    # Calculate stats
    if len(inference_times) == 0:
        print("No frames processed!")
        return
    
    avg_time = statistics.mean(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    median_time = statistics.median(inference_times)
    
    avg_fps = 1.0 / avg_time
    max_fps = 1.0 / min_time
    min_fps = 1.0 / max_time
    
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Frames measured: {len(inference_times)}")
    print(f"\nInference Time (per frame):")
    print(f"  Average: {avg_time*1000:.1f} ms")
    print(f"  Median:  {median_time*1000:.1f} ms")
    print(f"  Min:     {min_time*1000:.1f} ms")
    print(f"  Max:     {max_time*1000:.1f} ms")
    print(f"\nFPS:")
    print(f"  Average: {avg_fps:.1f} FPS")
    print(f"  Best:    {max_fps:.1f} FPS")
    print(f"  Worst:   {min_fps:.1f} FPS")
    
    # Estimate real-world with OpenCV overhead
    print(f"\nEstimated real-world FPS (with display/tracking): {avg_fps * 0.7:.1f} FPS")
    
    print("=" * 60)
    print("RECOMMENDATION:")
    if avg_fps >= 5:
        print("Good! CPU performance is sufficient for development.")
    elif avg_fps >= 3:
        print("Acceptable. May need to skip frames for real-time feel.")
    else:
        print("Slow. Consider using smaller input size or fewer frames.")
    print("=" * 60)

if __name__ == "__main__":
    benchmark()
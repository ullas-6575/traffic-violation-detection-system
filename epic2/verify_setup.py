# verify_setup.py
import torch
from ultralytics import YOLO
import time
import numpy as np

def check_setup():
    print("=" * 50)
    print("YOLOv8 Setup Verification")
    print("=" * 50)
    
    print(f"\nPython: {torch.__version__} (PyTorch)")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on: CPU")
    
    # Load nano model
    print("\nLoading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    
    print(f"Model loaded: {len(model.names)} classes")
    
    # Show vehicle classes
    vehicle_classes = {k: v for k, v in model.names.items() if v in ['car', 'motorcycle', 'bus', 'truck']}
    print(f"\nVehicle classes we'll track: {list(vehicle_classes.values())}")
    
    # Speed test
    print("\nSpeed test...")
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    start = time.time()
    results = model(dummy, verbose=False)
    elapsed = time.time() - start
    
    print(f"Inference time: {elapsed:.2f}s ({1/elapsed:.1f} FPS)")
    print("\n" + "=" * 50)
    print("Setup OK! Ready for vehicle detection.")
    print("=" * 50)

if __name__ == "__main__":
    check_setup()
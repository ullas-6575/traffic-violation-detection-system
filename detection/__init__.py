"""
Traffic Violation Detection System — core Python pipeline.

Combines YOLOv8 vehicle detection, ByteTrack tracking, bidirectional speed
estimation, red-light crossing detection, and Bangladesh license-plate
OCR (EasyOCR + Tesseract) into a single rule engine that emits structured
ViolationEvent records and writes them to the Laravel dashboard's database.

Entry point: `python -m detection.main` (see main.py for CLI flags).
"""

import os

# Keep scientific/CV libraries from over-allocating worker threads.
# Must be set before cv2/numpy/easyocr/torch are imported anywhere.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

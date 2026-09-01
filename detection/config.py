"""
Central configuration for the Traffic Violation Detection System.

All tunable parameters (camera source, detection zones, speed calibration,
signal timing, OCR thresholds, and asset paths) live here so the Raspberry
Pi deployment only ever needs to edit this one file.
"""

from pathlib import Path

# --- Project layout -------------------------------------------------------
# detection/config.py -> parents[0]=detection, parents[1]=project root
PROJECT_ROOT       = Path(__file__).resolve().parents[1]
ASSETS_DIR         = PROJECT_ROOT / "assets"
EVIDENCE_DIR        = ASSETS_DIR / "evidence"
EASYOCR_MODEL_DIR   = ASSETS_DIR / "models" / "easyocr"
TESSDATA_DIR        = ASSETS_DIR / "tessdata"
PLATE_MODEL_PATH    = ASSETS_DIR / "models" / "license_plate_detector.pt"

#  CONFIGURATION


# VIDEO_PATH accepts a file path, an RTSP URL (rtsp://user:pass@ip:554/stream)
# for an IP camera, or an integer-as-string ("0") for a USB webcam.
VIDEO_PATH           = "assets/videos/hd2.mp4"
MODEL_PATH           = str(ASSETS_DIR / "models" / "yolov8n.pt")
VEHICLE_CLASSES      = [2, 3, 5, 7]
CONFIDENCE           = 0.5
IOU                  = 0.45
DISPLAY_WIDTH        = 1280

# YOLO inference resolution. Default (640) is accurate but is the single
# biggest lever on Raspberry Pi CPU speed. Try 480 or 416 first if frames
# are lagging behind a live camera — each step down roughly trades a little
# small-object detection range for noticeably higher FPS. Must be a
# multiple of 32.
INFERENCE_IMGSZ      = 640

# --- Speed zone (TVS-7) ---
LINE_UPPER_Y         = 1200
LINE_LOWER_Y         = 1800
REAL_DISTANCE_METERS = 3.0
VIDEO_FPS            = 30.0
SPEED_LIMIT_KMH      = 60.0
MIN_FRAMES_VALID     = 3

# --- Stop line / Red light (TVS-8) ---
STOP_LINE_Y          = 1500

# --- Direction detection ---
DIRECTION_FRAMES     = 3       # Frames to observe before classifying direction
DIRECTION_MIN_MOVE   = 10      # Minimum Y pixel movement for classification

# --- Signal control ---
USE_KEYBOARD         = True
LIGHT_CYCLE_FRAMES   = 300
GREEN_FRAMES         = 120
YELLOW_FRAMES        = 60
RED_FRAMES           = 120

# --- Rule engine ---
COOLDOWN_FRAMES      = 90      # Min frames between two events for same (track, type)
EVIDENCE_PRE_FRAMES  = 15      # Frames before violation to include in clip window
EVIDENCE_POST_FRAMES = 30      # Frames after violation to include in clip window
MIN_TRACK_FRAMES     = 5       # Min frames tracked before red-light check
STALE_TRACK_FRAMES   = 100     # Frames after loss before track cleanup

# --- Plate recognition (TVS-10/11/12) ---
# (PROJECT_ROOT / EVIDENCE_DIR / EASYOCR_MODEL_DIR / TESSDATA_DIR / PLATE_MODEL_PATH
#  are defined at the top of this file)
OCR_CONFIDENCE_MIN   = 0.35
PLATE_DETECT_CONF    = 0.30
OCR_RETRY_FRAMES     = 15
PLATE_MIN_AREA_RATIO = 0.008
PLATE_MAX_AREA_RATIO = 0.35
PLATE_OCR_TOP_K      = 3       # OCR only the strongest localized crops per event
PLATE_MIN_RAW_WIDTH  = 24      # Reject tiny detections that contain no usable text
PLATE_MIN_RAW_HEIGHT = 10

# --- Ghost / occlusion ---
GHOST_FRAMES         = 10
REASSIGN_DIST        = 100




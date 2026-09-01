"""
plate_module/config.py
TVS-10: Configuration for license plate localization + OCR.
"""

import shutil

# --- Plate localization (OpenCV heuristic) ---
PLATE_MIN_AREA_PX       = 600
PLATE_MAX_AREA_RATIO    = 0.30
PLATE_ASPECT_MIN        = 3.0
PLATE_ASPECT_MAX        = 5.5
CANNY_LOW               = 50
CANNY_HIGH              = 150
BILATERAL_D             = 9
BILATERAL_SIGMA         = 75

# --- Tesseract OCR ---
TESSERACT_CMD = None
if TESSERACT_CMD is None:
    _tpath = shutil.which("tesseract")
    if _tpath:
        TESSERACT_CMD = _tpath

OCR_LANG                = "eng"
OCR_PSM                 = 7
OCR_WHITELIST           = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OCR_MIN_CONFIDENCE      = 35
OCR_MIN_TEXT_LEN        = 4
OCR_UPSCALE_FACTOR      = 2.5
OCR_USE_EASYOCR         = True
OCR_EASYOCR_LANGUAGES   = ["en"]
OCR_DOWNLOAD_MODELS     = True
OCR_UNREADABLE_VALUE    = "UNREADABLE"

# --- Pipeline behavior ---
PLATE_CROPS_DIR         = "evidence/plates"
RETRY_COOLDOWN_FRAMES   = 15

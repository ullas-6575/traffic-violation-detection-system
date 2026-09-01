"""Small launcher that checks for required packages and runs the app.

Usage: python run.py [--test] [--no-display]
"""
import sys

REQUIRED = ["cv2", "numpy", "supervision", "ultralytics", "pytesseract"]

missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except Exception:
        missing.append(pkg)

if missing:
    print("Missing required Python packages:", ", ".join(missing))
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

from app import main

if __name__ == "__main__":
    sys.exit(main())

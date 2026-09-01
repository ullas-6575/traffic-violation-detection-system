# TVS-9 + TVS-10 Project

## Quick start

1. Create a Python virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install Tesseract OCR on your system if you plan to use OCR:

Ubuntu/Debian:

```bash
sudo apt update && sudo apt install tesseract-ocr -y
```

4. Run the main app (uses `app.py`):

```bash
python app.py
```

Run unit tests:

```bash
python app.py --test
python violation_rule_engine.py --test
python -m pytest -q
```

## Notes

- The project expects video input at `VIDEO_PATH` (default is an IP camera URL). You can edit `violation_rule_engine.py` or pass a different video by modifying `app.py`.
- If you don't have the YOLO weights (`yolov8n.pt` and `models/license_plate_detector.pt`) the code will attempt to run with fallbacks, but detection quality will vary.
- If you run into missing package errors, install the required package shown in the error with `pip`.

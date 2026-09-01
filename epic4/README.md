# Epic 4 — Bangladesh Plate Recognition

`plate_recognition_pipeline.py` extends the Epic 3 violation engine with:

- TVS-10: plate localization, enhancement, and evidence image storage
- TVS-11: Bengali/English EasyOCR with Tesseract fallback
- TVS-12: confidence thresholding and `UNREADABLE` failure handling
- Vehicle body-color estimation with a confidence score for dashboard evidence

## Run

From the project root:

```powershell
$env:PYTHONUTF8 = "1"
python epic4\plate_recognition_pipeline.py
```

Run a Bangladesh traffic video without editing the source:

```powershell
python epic4\plate_recognition_pipeline.py --video videos\bangladesh_traffic.mp4
```

Controls: `R`, `Y`, `G` select the traffic light; `P` pauses; `Q` quits.

Run all Epic 3 and Epic 4 unit tests:

```powershell
python epic4\plate_recognition_pipeline.py --test
```

OCR a plate crop directly:

```powershell
python epic4\plate_recognition_pipeline.py --ocr-image path\to\plate.jpg
```

Evidence is written under `evidence/frames` and `evidence/plates`. The final
UTF-8 report is `violation_events.json`. EasyOCR models are project-local in
`models/easyocr`; Tesseract language files are in `tessdata`.

The supplied `models/license_plate_detector.pt` is used first for plate
localization. OpenCV contour localization remains available automatically when
the model finds no plausible plate. No additional model training is required.

For each violation, the runner now localizes plates across the tracked
pre-event frame window, ranks the actual plate crops by detector confidence and
visual quality, and sends only the best three crops to Bengali/English OCR. A
valid Bangladesh-style result is preferred over confident punctuation or other
OCR noise. Later retries only replace the saved evidence when they produce a
stronger result; arbitrary lower-bumper crops are no longer used as a fallback.

## Notes

The synced source folder remains under `epic4/files` for reference. Its large
duplicate videos are not copied into the project root; the combined runner uses
the existing root `videos` directory.

# AI Traffic Violation Detection System

An end-to-end computer-vision pipeline that watches a traffic camera feed,
detects and tracks vehicles, flags **overspeeding** and **red-light**
violations, crops and reads the vehicle's **license plate** (Bengali/English),
estimates the **vehicle body color**, and (optionally) writes every event to
a MySQL database for a Laravel dashboard.

Built incrementally as a series of "epics" — each folder is a development
stage, with **`epic4/plate_recognition_pipeline.py`** being the final,
combined, run-this-one script.

```
Camera / Video ──► YOLOv8 vehicle detection ──► ByteTrack tracking
                                                      │
                          ┌───────────────────────────┴───────────────────────┐
                          ▼                                                   ▼
                 Speed estimation (2-line zone)                Red-light crossing (stop line)
                          └───────────────────────────┬───────────────────────┘
                                                      ▼
                                        Violation Rule Engine (cooldown, dedup)
                                                      ▼
                          License-plate localization ──► EasyOCR (bn/en) ──► Tesseract fallback
                                                      ▼
                                Vehicle color estimation + evidence frames
                                                      ▼
                          evidence/frames, evidence/plates, violation_events.json
                                                      ▼
                                 (optional) MySQL + Laravel storage (epic5)
```

---

## 1. Repository structure

```
ISD/
├── epic2/                     # Stage 1-2: detection + tracking prototypes (dev/reference)
│   ├── detect_vehicles.py         YOLOv8 vehicle detection only
│   ├── track_vehicles.py          + ByteTrack persistent IDs
│   ├── track_lifecycle.py         track lifecycle / logging
│   ├── track_occlusion.py         occlusion handling
│   ├── track_persistent.py
│   ├── overlay_system.py          HUD/overlay drawing helpers
│   ├── benchmark_fps.py           CPU inference FPS benchmark
│   └── verify_setup.py            sanity-checks Torch/YOLO install
│
├── epic3/
│   └── violation_rule_engine.py   Speed + red-light rule engine (superseded by epic4)
│
├── epic4/                     # ★ FINAL, COMBINED PIPELINE — run this
│   ├── plate_recognition_pipeline.py   Detection + tracking + speed + red-light +
│   │                                    plate OCR + color estimation + reporting
│   ├── files/                     Original synced dev copy (kept for reference only;
│   │                               not imported by the pipeline above)
│   │   ├── app.py, plate_module/, videos/, ...
│   └── requirements.txt
│
├── epic5/                     # Optional MySQL / Laravel persistence layer
│   ├── db_writer.py               ViolationDBWriter — upserts events, queues on failure
│   └── requirements.txt
│
├── models/
│   ├── license_plate_detector.pt  Custom YOLO plate detector (bundled)
│   └── easyocr/                   EasyOCR model weights go here (empty — see §4.3)
├── tessdata/
│   ├── eng.traineddata            Tesseract English model (bundled)
│   └── ben.traineddata            Tesseract Bengali model (bundled)
├── yolov8n.pt                     COCO-pretrained YOLOv8-nano vehicle detector (bundled)
├── videos/                        Put your local test videos here
├── evidence/
│   ├── frames/                    Full-frame snapshot per violation
│   └── plates/                    Cropped/enhanced plate images
└── violation_events.json          Final run report (created on exit)
```

**Only `epic4/plate_recognition_pipeline.py` needs to be run.** Everything in
`epic2/` and `epic3/` are earlier development increments kept for reference;
`epic4/files/` is a duplicated source snapshot, also reference-only.

---

## 2. Features

- **Vehicle detection & tracking** — YOLOv8n (car/motorcycle/bus/truck) + a custom occlusion-aware ByteTrack wrapper for stable IDs.
- **Speed estimation** — two calibration lines + real-world distance → km/h.
- **Red-light violation detection** — configurable stop line, keyboard- or auto-cycled signal state (R/Y/G).
- **Violation rule engine** — per-track cooldown, structured `ViolationEvent` records, one event per (vehicle, violation type).
- **License-plate recognition (Bangladesh)** — custom YOLO plate localizer with an OpenCV contour fallback, best-of-3 crop selection across the pre-event frame window, Bengali+English **EasyOCR** as primary reader with **Tesseract** fallback, confidence thresholding and `UNREADABLE` handling.
- **Vehicle body-color estimation** with a confidence score.
- **Evidence capture** — full frame + plate crop saved per event.
- **JSON report** (`violation_events.json`) with every event's full metadata.
- **Optional MySQL/Laravel sync** (`epic5`) — upserts events and copies evidence images into a Laravel app's public storage; queues and retries on failure so the vision pipeline never blocks.
- **Built-in unit tests** (`--test` flag) and a **standalone OCR-on-image** utility (`--ocr-image`).

---

## 3. Requirements

### 3.1 Hardware
- Any machine with a webcam/USB camera, a video file, or network access to an IP camera.
- **Raspberry Pi 5 (4GB/8GB)** is supported for CPU-only inference — see the dedicated [Raspberry Pi 5 guide](#5-raspberry-pi-5-setup) below. Expect low single-digit to ~10 FPS with `yolov8n.pt`, which is fine for a fixed traffic-camera use case but not real-time 30 FPS.
- A monitor/desktop session (or VNC) — the app uses `cv2.imshow()` and keyboard input (`Q`/`P`/`R`/`Y`/`G`), so it is **not headless out of the box**.
- ~2 GB free disk for models + dependencies.

### 3.2 Software
- **Python 3.11** (also tested with 3.12).
- **Tesseract OCR** binary installed on the system (language files are bundled in `tessdata/`, but the `tesseract` executable itself must be installed separately).
- **MySQL Server** — only if you want the optional `epic5` database sync; otherwise the pipeline runs fine "offline" and just queues events locally.

### 3.3 Python packages

Combine everything into one `requirements.txt` at the project root:

```txt
# Core CV / detection / tracking
numpy
opencv-python>=4.8
ultralytics>=8.0
supervision>=0.18

# OCR
easyocr==1.7.2
pytesseract>=0.3.13

# Optional: MySQL persistence (epic5)
mysql-connector-python==9.7.0
```

> `ultralytics` pulls in PyTorch/TorchVision automatically. `easyocr` pulls in
> its own Torch, Pillow, scikit-image, Shapely, python-bidi, PyYAML, etc.
> On Windows keep `opencv-python` installed **last**, since `easyocr`
> declares the headless variant which otherwise overwrites `cv2.imshow`/`waitKey`
> (not an issue on Linux/Raspberry Pi OS).

---

## 4. Installation & running (desktop / Linux / Windows)

### 4.1 Clone/copy the project and create a virtual environment

```bash
cd ISD
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 4.2 Install Tesseract + Python dependencies

```bash
# Debian/Ubuntu/Raspberry Pi OS
sudo apt update && sudo apt install -y tesseract-ocr

# macOS
brew install tesseract

# Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
# and ensure the install folder is on PATH.

pip install -r requirements.txt
```

The bundled `tessdata/eng.traineddata` and `tessdata/ben.traineddata` are
picked up automatically by the pipeline (`TESSDATA_DIR`) — you do **not**
need to copy anything into the system Tesseract folder.

### 4.3 One-time EasyOCR model download

The pipeline loads EasyOCR with `download_enabled=False` and points it at
the local `models/easyocr/` folder (so a Pi with no/limited internet can
still run it). That folder ships **empty**, so on a machine that *does* have
internet, run this once to populate it:

```python
# run_once_download_easyocr_models.py
import easyocr
easyocr.Reader(
    ["bn", "en"],
    gpu=False,
    model_storage_directory="models/easyocr",
    download_enabled=True,
)
```

```bash
python run_once_download_easyocr_models.py
```

This downloads the CRAFT text detector plus the Bengali+English recognition
weights (a few hundred MB) into `models/easyocr/`. Copy that populated
folder onto the Raspberry Pi (or any offline machine) instead of
re-downloading there.

### 4.4 Add a video / camera source

Put test clips in `videos/`, or point straight at a live camera (see §6 for
IP-camera URLs). Default source is `videos/hd2.mp4` — override it with
`--video` (works for files **and** camera URLs, see §6).

### 4.5 Run it

From the project root:

```bash
python epic4/plate_recognition_pipeline.py
python epic4/plate_recognition_pipeline.py --video videos/traffic.mp4
python epic4/plate_recognition_pipeline.py --video "http://192.168.1.50:8080/video"   # IP camera
```

**Controls (while the window is focused):**
| Key | Action |
|---|---|
| `R` / `Y` / `G` | Manually set the traffic-light state (Red/Yellow/Green) |
| `P` | Pause / resume |
| `Q` | Quit and write the final report |

Other modes:

```bash
# Run the full unit-test suite
python epic4/plate_recognition_pipeline.py --test

# OCR a single cropped plate image, no video needed
python epic4/plate_recognition_pipeline.py --ocr-image path/to/plate.jpg
```

### 4.6 Output

- `evidence/frames/` — full annotated frame at the moment of each violation.
- `evidence/plates/` — cropped/enhanced plate image used for OCR.
- `violation_events.json` — final combined report: every `ViolationEvent`
  (event id, track id, violation type, speed, direction, signal state,
  plate number, OCR engine + confidence, vehicle color + confidence,
  evidence frame range) plus speed/red-light summary stats.

---

## 5. Raspberry Pi 5 setup

The Pi 5's Cortex-A76 CPU can run `yolov8n.pt` + the plate detector, but
there is **no GPU acceleration** for Torch/EasyOCR on stock Raspberry Pi OS —
everything runs on CPU. Plan for a few frames per second, which is adequate
for a fixed camera watching a stop line/speed zone.

### 5.1 OS & system packages

Use **Raspberry Pi OS Bookworm (64-bit)** — a 64-bit OS is required for
current PyTorch/Ultralytics wheels.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv python3-dev \
    tesseract-ocr libatlas-base-dev libjpeg-dev libopenjp2-7 \
    libgl1 libglib2.0-0 libtiff6 ffmpeg
```

`libgl1`/`libglib2.0-0` are needed for OpenCV's GUI build; `ffmpeg` improves
RTSP/HTTP camera-stream compatibility in `cv2.VideoCapture`.

### 5.2 Python environment

```bash
python3 -m venv ~/isd-venv
source ~/isd-venv/bin/activate
pip install --upgrade pip
```

### 5.3 Install PyTorch/Torch for ARM64 first

`ultralytics` and `easyocr` both need Torch. Install the CPU-only aarch64
wheel explicitly before the rest of `requirements.txt` (this avoids pip
trying to build Torch from source, which is very slow on a Pi):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If that index has no aarch64 build for your Python version, use the
community **piwheels** index instead (piwheels ships prebuilt ARM wheels for
Raspberry Pi):

```bash
pip install torch torchvision --extra-index-url https://www.piwheels.org/simple
```

### 5.4 Install the rest

```bash
pip install -r requirements.txt
```

### 5.5 Transfer the pre-downloaded EasyOCR models

Since the Pi may have slow/no reliable internet, download the EasyOCR
models on a desktop machine first (§4.3), then copy the populated folder
over:

```bash
scp -r models/easyocr pi@<pi-ip>:/home/pi/ISD/models/
```

### 5.6 Display: running the OpenCV window on a Pi

The pipeline uses `cv2.imshow` + keyboard controls, so you need one of:
- A monitor connected directly to the Pi 5, **or**
- A VNC session (`raspi-config` → enable VNC) so the desktop/window is
  visible remotely, **or**
- Run with X11 forwarding over SSH: `ssh -X pi@<pi-ip>`, then run the script
  (slower, but works for debugging).

Truly headless/no-display operation isn't supported out of the box — if you
need that, the small change is to remove the `cv2.imshow`/`cv2.waitKey`
calls in `epic4/plate_recognition_pipeline.py` and drive `paused`/quit via a
timer or an external signal instead.

### 5.7 Performance tips on Pi 5

- Keep `MODEL_PATH = "yolov8n.pt"` (do not switch to a larger YOLOv8 variant).
- Lower the incoming camera resolution/frame rate at the source if possible
  (e.g. 640×480 or 720p instead of 1080p+) — fewer pixels means faster
  YOLO + faster OCR crops.
- `DISPLAY_WIDTH` (top of `plate_recognition_pipeline.py`) only affects the
  preview window size, not detection resolution — lowering it saves a
  little `cv2.imshow`/resize overhead.
- If FPS is too low for your speed-zone calibration, measure your *actual*
  achieved processing FPS (printed on start, or benchmark with
  `epic2/benchmark_fps.py`) and set `VIDEO_FPS` in the config section to
  that measured value — the speed math assumes the frame rate you configure
  matches the frame rate you actually process at, not the source camera's
  nominal FPS.
- Consider exporting the YOLO models to **NCNN** or **ONNX** format
  (`model.export(format="ncnn")`) for a further CPU speed-up on ARM — this
  isn't wired into the script by default but the `.pt` files can be swapped
  for exported ones with matching `ultralytics` load code.

---

## 6. Using an IP camera as the video source

`cv2.VideoCapture()` accepts any URL, so `--video` works with a live network
camera exactly like a file path — no code changes needed.

### 6.1 Common URL formats

| Camera type | Example URL |
|---|---|
| RTSP camera (most CCTV/NVR cameras) | `rtsp://username:password@192.168.1.64:554/stream1` |
| MJPEG/HTTP camera | `http://192.168.1.64/video.mjpg` |
| Phone as IP camera — **IP Webcam** (Android) | `http://192.168.1.50:8080/video` |
| Phone as IP camera — **DroidCam** | `http://192.168.1.50:4747/video` |
| ONVIF/RTSP NVR channel | `rtsp://user:pass@192.168.1.10:554/cam/realmonitor?channel=1&subtype=0` |

```bash
python epic4/plate_recognition_pipeline.py --video "rtsp://admin:pass@192.168.1.64:554/stream1"
```

### 6.2 Notes for live streams

- Make sure the Raspberry Pi and the camera are on the **same network** and
  the port is reachable (`ping`/`curl` the camera's URL first to confirm).
- RTSP over Wi-Fi can be flaky; a wired connection or a strong 5GHz link is
  recommended for stable tracking.
- Because a live stream has no fixed length, the script will simply keep
  reading frames until the stream drops or you press `Q` — there's no
  "video ended" state to worry about.
- Re-calibrate `LINE_UPPER_Y`, `LINE_LOWER_Y`, `STOP_LINE_Y`, and
  `REAL_DISTANCE_METERS` (top of `plate_recognition_pipeline.py`) for your
  camera's actual mounting position/frame resolution — the values in the
  file were tuned for a specific test video's field of view.
- As noted in §5.7, set `VIDEO_FPS` to the frame rate you actually achieve
  end-to-end on the Pi, not the camera's advertised FPS, or speed
  calculations will be inaccurate.

---

## 7. Configuration reference

All tunables live at the top of `epic4/plate_recognition_pipeline.py`:

| Setting | Purpose |
|---|---|
| `VIDEO_PATH` | Default source if `--video` isn't passed |
| `MODEL_PATH` | Vehicle detector weights (`yolov8n.pt`) |
| `VEHICLE_CLASSES` | COCO class IDs kept: car(2), motorcycle(3), bus(5), truck(7) |
| `CONFIDENCE`, `IOU` | YOLO detection thresholds |
| `LINE_UPPER_Y`, `LINE_LOWER_Y`, `REAL_DISTANCE_METERS`, `VIDEO_FPS`, `SPEED_LIMIT_KMH` | Speed-zone calibration |
| `STOP_LINE_Y` | Red-light stop line position |
| `USE_KEYBOARD` | `True` = manual R/Y/G control, `False` = auto-cycle using `*_FRAMES` values |
| `COOLDOWN_FRAMES` | Minimum gap between two events for the same vehicle+violation type |
| `EVIDENCE_PRE_FRAMES` / `EVIDENCE_POST_FRAMES` | Frame window captured around each violation |
| `OCR_CONFIDENCE_MIN`, `PLATE_DETECT_CONF` | Plate/OCR acceptance thresholds |
| `PLATE_OCR_TOP_K` | How many candidate plate crops get OCR'd per event |

---

## 8. Optional: MySQL + Laravel sync (epic5)

If you also run the companion Laravel dashboard app, `epic4`'s main loop
automatically imports and uses `epic5.ViolationDBWriter`, which:

- Reads DB credentials from `app/.env` (Laravel's standard env file) at the
  project root — if that `app/` folder/`.env` doesn't exist, the writer
  simply reports `OFFLINE (queue enabled)` on startup and queues events to
  `epic5/pending_db_events.json` instead of failing.
- Upserts each `ViolationEvent` by its `event_id`.
- Copies full-frame and plate evidence images into
  `app/storage/app/public/violations`.
- Retries failed writes and flushes any previously queued events at the
  start of the next run.

```bash
# Test the DB connection
python epic5/db_writer.py --ping

# Backfill a previously generated report into the DB
python epic5/db_writer.py --import-json violation_events.json
```

```bash
# One-time, inside the Laravel app folder
cd app
php artisan storage:link
```

Install its extra dependency: `pip install mysql-connector-python==9.7.0`
(already included in the combined `requirements.txt` above).

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: Cannot open video: ...` | Check the path/URL, and that the camera is reachable (`ffplay <url>` or `curl -I <url>` first) |
| EasyOCR prints `EasyOCR unavailable: ...` and everything falls back to Tesseract | `models/easyocr/` is empty or incomplete — redo §4.3 |
| Very low FPS on Pi 5 | See §5.7; also confirm you installed the CPU aarch64 Torch wheel, not a source build |
| `cv2.imshow` window never appears / crashes on Pi | You're in a headless SSH session — enable VNC or connect a monitor (§5.6) |
| Tesseract errors (`TesseractNotFoundError`) | The `tesseract` binary isn't installed/on PATH — install via `apt`/`brew`/the Windows installer (§4.2) |
| Database shows `OFFLINE` | Expected if you haven't set up the Laravel `app/.env` — the pipeline still runs and just queues events locally (§8) |

---

## 10. Development history (epics)

- **Epic 2** — vehicle detection (`detect_vehicles.py`), ByteTrack tracking,
  occlusion handling, FPS benchmarking, HUD overlay prototypes.
- **Epic 3** — combined the speed and red-light detectors into a single
  `ViolationRuleEngine` with cooldown + structured events.
- **Epic 4** — final, all-in-one pipeline: adds plate localization, Bengali/
  English OCR (EasyOCR + Tesseract), vehicle color estimation, and the
  JSON report. **This is the script you actually run.**
- **Epic 5** — MySQL persistence + Laravel evidence-storage integration for
  a web dashboard, with offline queuing/retry.

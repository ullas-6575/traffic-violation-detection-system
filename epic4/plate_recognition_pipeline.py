"""
plate_recognition_pipeline.py
TVS-10/11/12: Bangladesh Plate Recognition — Epic 4 final module.

Combines speed estimation (TVS-7) and red light crossing detection (TVS-8)
into a single unified pipeline.

Extends the Epic 3 pipeline with violation-frame capture, license-plate
localization, Bengali/English OCR, confidence scoring, and graceful failure.

Subtasks:
1. Combine speed and red light checks in a single pipeline
2. Support configurable cooldown per track ID
3. Emit a structured ViolationEvent object with full metadata
4. Unit tests for edge cases

Design:
  - ViolationRuleEngine is a thin coordinator layer. It does NOT
    re-implement detection logic. It subscribes to events emitted
    by SpeedEstimator (TVS-7) and RedLightDetector (TVS-8) and
    merges them into canonical ViolationEvents.
  - One ViolationEvent per (track_id, violation_type) incident.
  - Cooldown is enforced per (track_id, type) pair so a speeding
    vehicle that also runs a red light gets TWO separate events.
  - Evidence clip: the engine records the frame window around each
    violation for downstream plate-crop and OCR modules.
"""

from __future__ import annotations

import os

# Keep scientific/CV libraries from over-allocating worker threads on Windows.
# This must be set before importing cv2/numpy/easyocr/scipy-backed packages.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import supervision as sv
from ultralytics import YOLO
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict
import json
import math
import re
import unittest
from unittest import mock
from collections import deque
from datetime import datetime
from pathlib import Path



#  CONFIGURATION


VIDEO_PATH           = "videos/hd2.mp4"
MODEL_PATH           = "yolov8n.pt"
VEHICLE_CLASSES      = [2, 3, 5, 7]
CONFIDENCE           = 0.5
IOU                  = 0.45
DISPLAY_WIDTH        = 1280

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
PROJECT_ROOT         = Path(__file__).resolve().parents[1]
EVIDENCE_DIR         = PROJECT_ROOT / "evidence"
EASYOCR_MODEL_DIR    = PROJECT_ROOT / "models" / "easyocr"
TESSDATA_DIR         = PROJECT_ROOT / "tessdata"
PLATE_MODEL_PATH     = PROJECT_ROOT / "models" / "license_plate_detector.pt"
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



#  DATA MODEL


class Direction(Enum):
    UNKNOWN = "UNKNOWN"
    UP      = "UP"       # Y decreasing — enters from bottom
    DOWN    = "DOWN"     # Y increasing — enters from top


class SignalState(Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    RED    = "RED"


class ViolationType(str, Enum):
    OVERSPEED  = "OVERSPEED"
    RED_LIGHT  = "RED_LIGHT"


@dataclass
class SpeedMeasurement:
    """Single speed measurement result (TVS-7)."""
    track_id: int
    direction: str
    speed_kmh: float
    violation: bool
    start_frame: int
    end_frame: int
    frames_between: float
    time_seconds: float


@dataclass
class ViolationEvent:
    """
    Canonical violation record emitted by the rule engine.
    Consumed downstream by: plate crop → OCR → MySQL writer → Laravel dashboard.
    """
    event_id:             str              # unique: "{track_id}_{type}_{frame}_{counter}"
    track_id:             int
    violation_type:       ViolationType
    frame_number:         int
    timestamp:            str              # ISO 8601
    direction:            str              # "up" / "down" / "UP" / "DOWN" / "unknown"
    signal_state:         str              # "RED" / "GREEN" / "YELLOW" / "N/A"
    speed_kmh:            Optional[float]  # None for red-light-only events
    speed_limit_kmh:      Optional[float]
    bbox:                 list             # [x1, y1, x2, y2] at violation frame
    evidence_start_frame: int              # clip window start
    evidence_end_frame:   int              # clip window end
    plate_number:         str = ""         # filled by TVS-10/11
    image_path:           str = ""         # filled by TVS-10
    plate_crop_path:      str = ""         # cropped/enhanced plate evidence
    ocr_raw_text:         str = ""         # unmodified winning OCR output
    ocr_confidence:       float = 0.0       # normalized 0..1
    ocr_engine:           str = ""         # easyocr / tesseract / none
    vehicle_color:        str = "UNKNOWN"  # estimated body color
    color_confidence:     float = 0.0       # normalized 0..1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["violation_type"] = self.violation_type.value
        return d



#  TVS-10/11/12: BANGLADESH PLATE RECOGNITION


@dataclass
class OCRCandidate:
    text: str
    confidence: float
    engine: str


class BangladeshPlateRecognizer:
    """Locate and read Bengali/English plates inside a violating vehicle ROI."""

    _BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

    def __init__(self, confidence_min: float = OCR_CONFIDENCE_MIN,
                 lazy: bool = True):
        self.confidence_min = confidence_min
        self._reader = None
        self._plate_model = None
        self._plate_model_checked = False
        self._easyocr_error = ""
        self._tesseract_error = ""
        self._event_best_scores: Dict[str, float] = {}
        (EVIDENCE_DIR / "frames").mkdir(parents=True, exist_ok=True)
        (EVIDENCE_DIR / "plates").mkdir(parents=True, exist_ok=True)
        EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if not lazy:
            self._get_easyocr_reader()

    def _get_plate_model(self):
        """Load the supplied trained plate detector once, with safe fallback."""
        if self._plate_model_checked:
            return self._plate_model
        self._plate_model_checked = True
        if not PLATE_MODEL_PATH.exists():
            print(f"  [PLATE] Model missing: {PLATE_MODEL_PATH}; using OpenCV fallback")
            return None
        try:
            self._plate_model = YOLO(str(PLATE_MODEL_PATH))
            print(f"  [PLATE] Loaded trained detector: {PLATE_MODEL_PATH.name}")
        except Exception as exc:
            print(f"  [PLATE] Model failed to load: {exc}; using OpenCV fallback")
        return self._plate_model

    def _get_easyocr_reader(self):
        if self._reader is not None:
            return self._reader
        if self._easyocr_error:
            return None
        try:
            import easyocr
            self._reader = easyocr.Reader(
                ["bn", "en"], gpu=False,
                model_storage_directory=str(EASYOCR_MODEL_DIR),
                download_enabled=False, verbose=False,
            )
        except Exception as exc:
            self._easyocr_error = str(exc)
            print(f"  [OCR] EasyOCR unavailable: {exc}")
        return self._reader

    @staticmethod
    def _safe_bbox(frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    @classmethod
    def estimate_vehicle_color(cls, frame, bbox):
        """Estimate a coarse body color from the central vehicle region."""
        x1, y1, x2, y2 = cls._safe_bbox(frame, bbox)
        vehicle = frame[y1:y2, x1:x2]
        if vehicle.size == 0 or vehicle.shape[0] < 12 or vehicle.shape[1] < 12:
            return "UNKNOWN", 0.0

        h, w = vehicle.shape[:2]
        # Central body region reduces road, windows, lights and plate influence.
        body = vehicle[int(h * 0.20):int(h * 0.72),
                       int(w * 0.12):int(w * 0.88)]
        if body.size == 0:
            return "UNKNOWN", 0.0
        body = cv2.resize(body, (80, 60), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        labels = {
            "BLACK": val < 55,
            "WHITE": (sat < 42) & (val >= 185),
            "GRAY": (sat < 55) & (val >= 55) & (val < 185),
            "RED": (sat >= 55) & (val >= 55) & ((hue < 10) | (hue >= 170)),
            "ORANGE": (sat >= 55) & (val >= 55) & (hue >= 10) & (hue < 22),
            "YELLOW": (sat >= 55) & (val >= 55) & (hue >= 22) & (hue < 35),
            "GREEN": (sat >= 55) & (val >= 45) & (hue >= 35) & (hue < 85),
            "BLUE": (sat >= 55) & (val >= 45) & (hue >= 85) & (hue < 135),
            "PURPLE": (sat >= 55) & (val >= 45) & (hue >= 135) & (hue < 170),
        }
        counts = {name: int(mask.sum()) for name, mask in labels.items()}
        color, count = max(counts.items(), key=lambda item: item[1])
        total = body.shape[0] * body.shape[1]
        confidence = count / float(total)
        if confidence < 0.18:
            return "UNKNOWN", round(confidence, 4)
        return color, round(confidence, 4)

    @staticmethod
    def _plate_visual_score(crop, detector_confidence: float = 0.0) -> float:
        """Score whether a raw crop contains plate-like, OCR-usable detail."""
        if crop is None or crop.size == 0:
            return -1.0
        h, w = crop.shape[:2]
        if w < PLATE_MIN_RAW_WIDTH or h < PLATE_MIN_RAW_HEIGHT:
            return -1.0
        aspect = w / float(max(1, h))
        # Bangladesh plates may be one or two lines, so allow both layouts.
        if not (0.9 <= aspect <= 7.5):
            return -1.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        contrast = float(gray.std())
        edge_density = cv2.countNonZero(cv2.Canny(gray, 50, 150)) / float(gray.size)
        if contrast < 10.0 or not (0.015 <= edge_density <= 0.60):
            return -1.0

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edge_score = max(0.0, 1.0 - abs(edge_density - 0.18) / 0.18)
        return (
            detector_confidence * 3.0
            + min(contrast / 55.0, 1.0)
            + edge_score
            + min(sharpness / 500.0, 1.0)
            + min(h / 40.0, 1.0)
        )

    def locate_plate_candidate(self, frame, bbox):
        """Return the strongest plate crop and its localization-quality score."""
        x1, y1, x2, y2 = self._safe_bbox(frame, bbox)
        vehicle = frame[y1:y2, x1:x2]
        if vehicle.size == 0 or vehicle.shape[0] < 20 or vehicle.shape[1] < 30:
            return None

        vh, vw = vehicle.shape[:2]

        model = self._get_plate_model()
        if model is not None:
            try:
                results = model(vehicle, verbose=False, conf=PLATE_DETECT_CONF)
                boxes = results[0].boxes if results else None
                if boxes is not None and len(boxes):
                    yolo_candidates = []
                    for box in boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        pad_x = max(2, int((bx2 - bx1) * 0.06))
                        pad_y = max(2, int((by2 - by1) * 0.12))
                        bx1 = max(0, int(bx1) - pad_x)
                        by1 = max(0, int(by1) - pad_y)
                        bx2 = min(vw, int(bx2) + pad_x)
                        by2 = min(vh, int(by2) + pad_y)
                        candidate = vehicle[by1:by2, bx1:bx2]
                        confidence = float(box.conf[0].cpu())
                        score = self._plate_visual_score(candidate, confidence)
                        if score >= 0:
                            yolo_candidates.append((score, candidate.copy()))
                    if yolo_candidates:
                        return max(yolo_candidates, key=lambda item: item[0])
            except Exception as exc:
                print(f"  [PLATE] YOLO inference failed; using OpenCV fallback: {exc}")

        search_y = int(vh * 0.35)
        search = vehicle[search_y:, :]
        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), iterations=2,
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)

        search_area = float(search.shape[0] * search.shape[1])
        best = None
        best_score = -1.0
        for contour in contours:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            if rh == 0:
                continue
            ratio = rw / float(rh)
            area_ratio = (rw * rh) / search_area
            if not (1.4 <= ratio <= 6.5):
                continue
            if not (PLATE_MIN_AREA_RATIO <= area_ratio <= PLATE_MAX_AREA_RATIO):
                continue
            rectangularity = cv2.contourArea(contour) / max(1.0, rw * rh)
            position = (ry + rh / 2) / search.shape[0]
            score = rectangularity + min(ratio, 4.0) / 4.0 + position * 0.35
            if score > best_score:
                best_score = score
                best = (rx, ry, rw, rh)

        if best is None:
            # Do not OCR an arbitrary bumper region; a clean failure is safer.
            return None

        rx, ry, rw, rh = best
        px, py = max(3, int(rw * 0.06)), max(3, int(rh * 0.18))
        sx1, sy1 = max(0, rx - px), max(0, search_y + ry - py)
        sx2, sy2 = min(vw, rx + rw + px), min(vh, search_y + ry + rh + py)
        crop = vehicle[sy1:sy2, sx1:sx2]
        visual_score = self._plate_visual_score(crop)
        return (visual_score, crop.copy()) if visual_score >= 0 else None

    def locate_plate(self, frame, bbox):
        """Backward-compatible crop-only plate localization API."""
        candidate = self.locate_plate_candidate(frame, bbox)
        return candidate[1] if candidate is not None else None

    @staticmethod
    def preprocess(crop):
        scale = max(2.0, 320.0 / max(1, crop.shape[1]))
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        denoised = cv2.bilateralFilter(clahe, 7, 45, 45)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 9,
        )
        return enlarged, denoised, binary

    @classmethod
    def normalize_text(cls, text: str) -> str:
        text = text.translate(cls._BN_DIGITS).upper()
        text = text.replace("|", "").replace("_", "-")
        # Keep Bengali letters, ASCII letters/digits, spaces and separators.
        text = re.sub(r"[^\u0980-\u09FFA-Z0-9\-\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip(" -")
        return text

    @staticmethod
    def is_plausible_plate(text: str) -> bool:
        """Reject fragments that cannot be a complete Bangladesh plate."""
        compact = re.sub(r"[\s-]", "", text)
        digits = sum(ch.isdigit() for ch in compact)
        bengali_letters = sum("\u0980" <= ch <= "\u09ff" and ch.isalpha()
                              for ch in compact)
        return len(compact) >= 6 and digits >= 4 and bengali_letters >= 1

    def _run_easyocr(self, images) -> Optional[OCRCandidate]:
        reader = self._get_easyocr_reader()
        if reader is None:
            return None
        candidates = []
        try:
            for image in images:
                results = reader.readtext(image, detail=1, paragraph=False)
                if not results:
                    continue
                text = " ".join(str(item[1]) for item in results).strip()
                weights = [max(1, len(str(item[1]))) for item in results]
                confidence = sum(float(item[2]) * weight
                                 for item, weight in zip(results, weights)) / sum(weights)
                candidates.append(OCRCandidate(text, confidence, "easyocr"))
        except Exception as exc:
            self._easyocr_error = str(exc)
            print(f"  [OCR] EasyOCR failed: {exc}")
        return max(candidates, key=lambda c: c.confidence) if candidates else None

    def _run_tesseract(self, images) -> Optional[OCRCandidate]:
        try:
            import pytesseract
            from pytesseract import Output
            candidates = []
            languages = "ben+eng" if (TESSDATA_DIR / "ben.traineddata").exists() else "eng"
            # Forward slashes avoid pytesseract/Tesseract splitting a quoted
            # Windows path into `"directory"/language.traineddata`.
            config = f"--tessdata-dir {TESSDATA_DIR.as_posix()} --psm 6"
            for image in images:
                data = pytesseract.image_to_data(
                    image, lang=languages, config=config, output_type=Output.DICT,
                )
                pairs = [(str(t).strip(), float(c))
                         for t, c in zip(data["text"], data["conf"])
                         if str(t).strip() and float(c) >= 0]
                if not pairs:
                    continue
                text = " ".join(t for t, _ in pairs)
                confidence = sum(c * max(1, len(t)) for t, c in pairs) / sum(
                    max(1, len(t)) for t, _ in pairs)
                candidates.append(OCRCandidate(text, confidence / 100.0, "tesseract"))
            return max(candidates, key=lambda c: c.confidence) if candidates else None
        except Exception as exc:
            self._tesseract_error = str(exc)
            print(f"  [OCR] Tesseract failed: {exc}")
            return None

    def recognize(self, crop) -> OCRCandidate:
        enlarged, enhanced, binary = self.preprocess(crop)
        easy = self._run_easyocr([enlarged, enhanced])
        easy_cleaned = self.normalize_text(easy.text) if easy else ""
        if (easy and easy.confidence >= self.confidence_min
                and self.is_plausible_plate(easy_cleaned)):
            return easy
        tess = self._run_tesseract([enhanced, binary])
        candidates = [c for c in (easy, tess) if c is not None]
        if not candidates:
            return OCRCandidate("", 0.0, "none")
        return max(
            candidates,
            key=lambda c: c.confidence + (
                1.0 if self.is_plausible_plate(self.normalize_text(c.text)) else 0.0
            ),
        )

    def process_event_candidates(self, frame_candidates, event: ViolationEvent) -> ViolationEvent:
        """Localize across frames, then OCR only the strongest plate crops."""
        localized = []
        for frame, bbox in frame_candidates:
            candidate = self.locate_plate_candidate(frame, bbox)
            if candidate is not None:
                localization_score, crop = candidate
                localized.append((localization_score, frame, bbox, crop))

        if not localized:
            if event.event_id not in self._event_best_scores:
                frame, bbox = frame_candidates[-1]
                color, color_confidence = self.estimate_vehicle_color(frame, bbox)
                event.vehicle_color = color
                event.color_confidence = color_confidence
                frame_path = EVIDENCE_DIR / "frames" / f"{event.event_id}_frame.jpg"
                cv2.imwrite(str(frame_path), frame)
                event.image_path = str(frame_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                event.plate_number = "UNREADABLE"
                event.ocr_engine = "none"
                self._event_best_scores[event.event_id] = -1.0
            return event

        localized.sort(key=lambda item: item[0], reverse=True)
        evaluated = []
        for localization_score, frame, bbox, crop in localized[:PLATE_OCR_TOP_K]:
            result = self.recognize(crop)
            cleaned = self.normalize_text(result.text)
            valid = (
                result.confidence >= self.confidence_min
                and self.is_plausible_plate(cleaned)
            )
            # A plausible full plate must outrank confident OCR punctuation/noise.
            final_score = localization_score + result.confidence * 2.0 + (5.0 if valid else 0.0)
            evaluated.append((final_score, valid, cleaned, result, frame, bbox, crop))

        best = max(evaluated, key=lambda item: item[0])
        final_score, valid, cleaned, result, frame, bbox, crop = best
        if final_score <= self._event_best_scores.get(event.event_id, float("-inf")):
            return event
        self._event_best_scores[event.event_id] = final_score

        color, color_confidence = self.estimate_vehicle_color(frame, bbox)
        if color_confidence >= event.color_confidence:
            event.vehicle_color = color
            event.color_confidence = color_confidence

        frame_path = EVIDENCE_DIR / "frames" / f"{event.event_id}_frame.jpg"
        cv2.imwrite(str(frame_path), frame)
        event.image_path = str(frame_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        _, enhanced, _ = self.preprocess(crop)
        plate_path = EVIDENCE_DIR / "plates" / f"{event.event_id}_plate.jpg"
        cv2.imwrite(str(plate_path), enhanced)
        event.plate_crop_path = str(plate_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

        event.ocr_raw_text = result.text
        event.ocr_confidence = round(float(result.confidence), 4)
        event.ocr_engine = result.engine
        event.plate_number = cleaned if valid else "UNREADABLE"
        print(f"  [OCR] {event.event_id}: {event.plate_number} "
              f"({event.ocr_engine}, {event.ocr_confidence:.1%})")
        return event

    def process_event(self, frame, event: ViolationEvent,
                      evidence_bbox=None) -> ViolationEvent:
        """Process one frame while preserving the best result across retries."""
        return self.process_event_candidates(
            [(frame, evidence_bbox or event.bbox)], event
        )


#  TRAFFIC LIGHT


class TrafficLight:
    def __init__(self):
        self.state = SignalState.GREEN

    def set_state(self, key: int):
        if key == ord('r'):
            self.state = SignalState.RED
            print("  Signal → RED")
        elif key == ord('y'):
            self.state = SignalState.YELLOW
            print("  Signal → YELLOW")
        elif key == ord('g'):
            self.state = SignalState.GREEN
            print("  Signal → GREEN")

    def auto_update(self, frame_num: int):
        pos = frame_num % LIGHT_CYCLE_FRAMES
        if pos < GREEN_FRAMES:
            self.state = SignalState.GREEN
        elif pos < GREEN_FRAMES + YELLOW_FRAMES:
            self.state = SignalState.YELLOW
        else:
            self.state = SignalState.RED

    def get_bgr(self) -> tuple:
        return {
            SignalState.GREEN:  (0, 255, 0),
            SignalState.YELLOW: (0, 255, 255),
            SignalState.RED:    (0, 0, 255),
        }[self.state]



#  TVS-7: SPEED ESTIMATOR
#  History buffer + sub-frame interpolation for high-speed accuracy.


class SpeedEstimator:
    """
    Advanced speed estimator with a Coordinate History Buffer
    and Sub-Frame Interpolation for high-speed accuracy.
    Ported from TVS-7 (speed_estimation_bidirectional.py).
    """

    def __init__(self, line_upper: int, line_lower: int,
                 real_distance_m: float, fps: float, speed_limit: float):
        self.line_upper       = line_upper
        self.line_lower       = line_lower
        self.real_distance_m  = real_distance_m
        self.fps              = fps
        self.speed_limit      = speed_limit
        self.pixel_distance   = abs(line_lower - line_upper)
        self.pixels_per_meter = self.pixel_distance / real_distance_m

        self._tracks: Dict[int, dict] = {}
        self._ghosts: List[dict]      = []
        self.measurements: List[SpeedMeasurement] = []
        self.discarded_count = 0

    def _get_track(self, track_id: int) -> dict:
        """Get or create track state using a history buffer."""
        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "state": "active",
                "direction": None,
                "history": [],  # (frame_num, top_y, bottom_y, center_y)
            }
        return self._tracks[track_id]

    def _find_ghost_match(self, cx: float, cy: float) -> Optional[dict]:
        best_match = None
        best_dist = float('inf')
        for ghost in self._ghosts:
            if ghost["frames_since_lost"] > GHOST_FRAMES:
                continue
            gx, gy = ghost["last_pos"]
            dist = math.sqrt((cx - gx)**2 + (cy - gy)**2)
            if dist < REASSIGN_DIST and dist < best_dist:
                best_dist = dist
                best_match = ghost
        return best_match

    def _ghost_track(self, track_id: int, track: dict):
        if len(track["history"]) > 0:
            last_cx_cy = (0, track["history"][-1][3])
            self._ghosts.append({
                "last_pos": last_cx_cy,
                "state": track.copy(),
                "frames_since_lost": 0,
            })

    def process(self, detections: sv.Detections, frame_num: int) -> List[dict]:
        """
        Process detections for speed estimation.
        Returns list of speed event dicts (with bbox for rule engine).
        """
        events = []

        if detections.tracker_id is None:
            for tid, track in list(self._tracks.items()):
                self._ghost_track(tid, track)
            self._tracks.clear()
            return events

        tracker_ids = detections.tracker_id
        current_ids = set(tracker_ids.tolist() if hasattr(tracker_ids, 'tolist') else tracker_ids)
        lost_ids = set(self._tracks.keys()) - current_ids

        for tid in lost_ids:
            self._ghost_track(tid, self._tracks[tid])
            del self._tracks[tid]

        tracker_ids = detections.tracker_id
        for i, track_id in enumerate(tracker_ids.tolist() if hasattr(tracker_ids, 'tolist') else tracker_ids):
            x1, y1, x2, y2 = detections.xyxy[i]
            top_y, bottom_y = float(y1), float(y2)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if track_id not in self._tracks:
                ghost = self._find_ghost_match(cx, cy)
                if ghost:
                    self._tracks[track_id] = ghost["state"].copy()
                    self._ghosts.remove(ghost)
                else:
                    self._get_track(track_id)

            track = self._tracks[track_id]

            # 1. ADD TO HISTORY BUFFER
            if track["state"] == "active":
                track["history"].append((frame_num, top_y, bottom_y, cy))

                # 2. EVALUATE THE BUFFER
                event = self._evaluate_history(
                    track, track_id,
                    bbox=[float(x1), float(y1), float(x2), float(y2)]
                )
                if event:
                    events.append(event)

        # Clean old ghosts
        self._ghosts = [g for g in self._ghosts if g["frames_since_lost"] <= GHOST_FRAMES]
        for g in self._ghosts:
            g["frames_since_lost"] += 1

        return events

    def _get_exact_crossing_frame(self, history: list, line_y: float,
                                   is_up: bool, edge_idx: int) -> Optional[float]:
        """Calculates the exact sub-frame a line was crossed using linear interpolation."""
        for i in range(1, len(history)):
            prev_f = history[i-1][0]
            curr_f = history[i][0]

            y_prev = history[i-1][edge_idx]
            y_curr = history[i][edge_idx]

            if is_up:  # Moving up: Y is decreasing
                if y_prev > line_y >= y_curr:
                    ratio = (y_prev - line_y) / (y_prev - y_curr + 1e-6)
                    return prev_f + ratio * (curr_f - prev_f)
            else:      # Moving down: Y is increasing
                if y_prev < line_y <= y_curr:
                    ratio = (line_y - y_prev) / (y_curr - y_prev + 1e-6)
                    return prev_f + ratio * (curr_f - prev_f)
        return None

    def _evaluate_history(self, track: dict, track_id: int,
                          bbox: list = None) -> Optional[dict]:
        history = track["history"]

        # --- Step A: Determine Direction ---
        if track["direction"] is None:
            if len(history) >= DIRECTION_FRAMES:
                first_y = history[0][3]
                last_y = history[-1][3]
                diff = last_y - first_y

                if abs(diff) < 5:
                    if len(history) > 90:
                        track["state"] = "discarded"
                        self.discarded_count += 1
                    return None

                track["direction"] = "up" if diff < 0 else "down"
            return None

        # --- Step B: Check History Buffer for Crossings ---
        direction = track["direction"]

        if direction == "up":
            # Edge index 1 is top_y
            start_frame_exact = self._get_exact_crossing_frame(
                history, self.line_lower, True, 1)
            end_frame_exact = self._get_exact_crossing_frame(
                history, self.line_upper, True, 1)
        else:
            # Edge index 2 is bottom_y
            start_frame_exact = self._get_exact_crossing_frame(
                history, self.line_upper, False, 2)
            end_frame_exact = self._get_exact_crossing_frame(
                history, self.line_lower, False, 2)

        # --- Step C: Calculate Speed ---
        if start_frame_exact is not None and end_frame_exact is not None:
            frames = end_frame_exact - start_frame_exact

            if frames <= 0 or frames < MIN_FRAMES_VALID:
                track["state"] = "discarded"
                self.discarded_count += 1
                return None

            time_s = frames / self.fps
            speed_ms = self.real_distance_m / time_s
            speed_kmh = round(speed_ms * 3.6, 1)
            violation = speed_kmh > self.speed_limit

            measurement = SpeedMeasurement(
                track_id=track_id,
                direction=direction,
                speed_kmh=speed_kmh,
                violation=violation,
                start_frame=int(start_frame_exact),
                end_frame=int(end_frame_exact),
                frames_between=round(frames, 2),
                time_seconds=round(time_s, 3),
            )

            self.measurements.append(measurement)
            track["state"] = "done"
            track["history"] = []  # Clear memory

            return {
                "track_id":   track_id,
                "direction":  direction,
                "speed_kmh":  speed_kmh,
                "violation":  violation,
                "frames":     round(frames, 2),
                "time_s":     round(time_s, 3),
                "frame_num":  int(end_frame_exact),
                "bbox":       bbox or [0, 0, 0, 0],
            }

        return None

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_speed(self, track_id: int) -> Optional[float]:
        for m in self.measurements:
            if m.track_id == track_id:
                return m.speed_kmh
        return None

    def is_violation(self, track_id: int) -> bool:
        for m in self.measurements:
            if m.track_id == track_id:
                return m.violation
        return False

    def get_track_state(self, track_id: int) -> str:
        if track_id in self._tracks:
            return self._tracks[track_id]["state"]
        return "unknown"

    def get_direction(self, track_id: int) -> Optional[str]:
        if track_id in self._tracks:
            return self._tracks[track_id].get("direction")
        return None

    def get_summary(self) -> dict:
        total = len(self.measurements)
        violations = sum(1 for m in self.measurements if m.violation)
        avg_speed = sum(m.speed_kmh for m in self.measurements) / total if total else 0
        up = sum(1 for m in self.measurements if m.direction == "up")
        down = sum(1 for m in self.measurements if m.direction == "down")

        return {
            "total_valid": total,
            "up_count": up,
            "down_count": down,
            "discarded": self.discarded_count,
            "violations": violations,
            "average_speed": round(avg_speed, 1),
            "speed_limit": self.speed_limit,
            "real_distance_m": self.real_distance_m,
            "pixels_per_meter": round(self.pixels_per_meter, 1),
            "measurements": [
                {
                    "track_id": m.track_id,
                    "direction": m.direction,
                    "speed_kmh": m.speed_kmh,
                    "violation": m.violation,
                    "frames": m.frames_between,
                    "time_s": m.time_seconds,
                }
                for m in self.measurements
            ]
        }



#  TVS-8: RED LIGHT DETECTOR
#  Bidirectional, history buffer, retroactive check, signal-per-frame fix.


class RedLightDetector:
    """
    Red light violation detector for bidirectional traffic.
    Uses a Coordinate History Buffer to catch fast vehicles that cross
    the stop line before their direction is classified (retroactive check).

    Signal-per-frame fix: stores the signal state alongside each history
    entry so retroactive checks use the signal AT THE TIME OF CROSSING,
    not the current frame's signal.

    Ported from TVS-8 (red_light_bi.py).
    """

    def __init__(self, stop_line_y: int):
        self.stop_line_y = stop_line_y
        self.tracks:     dict = {}
        self.violations: list = []

    # ── Track management ──────────────────────────────────────────────────────

    def _init_track(self, track_id: int, frame_num: int):
        self.tracks[track_id] = {
            "direction":       Direction.UNKNOWN,
            "history":         [],  # (frame_num, top_y, bottom_y, center_y, signal_state)
            "last_edge_y":     None,
            "frames_tracked":  0,
            "last_seen_frame": frame_num,
            "violated":        False,
            "violation_frame": None,
        }

    def _classify_direction(self, history: list) -> Direction:
        """
        Compare first and last center_y in the history buffer.
        Negative delta → moving UP. Positive delta → moving DOWN.
        """
        if len(history) < DIRECTION_FRAMES:
            return Direction.UNKNOWN

        # History tuples: (frame_num, top_y, bottom_y, center_y, signal_state)
        # Index 3 is center_y
        delta = history[-1][3] - history[0][3]
        if abs(delta) < DIRECTION_MIN_MOVE:
            return Direction.UNKNOWN
        return Direction.UP if delta < 0 else Direction.DOWN

    def _retroactive_check(self, track: dict) -> Optional[dict]:
        """
        Scan history buffer for a crossing that happened during the
        direction classification blind spot.
        Uses the signal state STORED AT THAT FRAME — not the current signal.
        """
        direction = track["direction"]
        for i in range(1, len(track["history"])):
            pf, pt, pb, _, ps = track["history"][i-1]
            cf, ct, cb, _, cs = track["history"][i]
            # Only flag if signal was RED at the time of crossing
            if cs != SignalState.RED:
                continue
            if direction == Direction.UP and pt > self.stop_line_y >= ct:
                return {"frame": cf, "edge_label": "top_y",
                        "edge_val": ct, "signal": cs.value}
            if direction == Direction.DOWN and pb < self.stop_line_y <= cb:
                return {"frame": cf, "edge_label": "bottom_y",
                        "edge_val": cb, "signal": cs.value}
        return None

    # ── Main update ───────────────────────────────────────────────────────────

    def process(self, detections: sv.Detections,
                signal: SignalState, frame_num: int) -> list:
        """
        Process detections for red light violations.
        Returns list of violation event dicts (with bbox for rule engine).
        """
        events     = []
        active_ids = set()

        if detections.tracker_id is None:
            self._cleanup(frame_num, active_ids)
            return events

        active_ids = set(int(tid) for tid in detections.tracker_id)

        for i, track_id in enumerate(detections.tracker_id):
            track_id = int(track_id)
            x1, y1, x2, y2 = detections.xyxy[i]
            top_y    = float(y1)
            bottom_y = float(y2)
            center_y = (top_y + bottom_y) / 2
            bbox     = [float(x1), float(y1), float(x2), float(y2)]

            # Init new track
            if track_id not in self.tracks:
                self._init_track(track_id, frame_num)

            track = self.tracks[track_id]
            track["frames_tracked"]  += 1
            track["last_seen_frame"]  = frame_num

            # 1. ALWAYS ADD TO HISTORY BUFFER (with signal state)
            track["history"].append((frame_num, top_y, bottom_y, center_y, signal))

            # 2. CLASSIFY DIRECTION
            if track["direction"] == Direction.UNKNOWN:
                track["direction"] = self._classify_direction(track["history"])

                # The exact moment direction is found:
                if track["direction"] != Direction.UNKNOWN:
                    print(f"  Track #{track_id} classified as {track['direction'].value}")
                    track["last_edge_y"] = (top_y if track["direction"] == Direction.UP
                                            else bottom_y)

                    # 3. RUN RETROACTIVE CHECK ON THE BUFFER
                    retro = self._retroactive_check(track)
                    if retro and not track["violated"]:
                        track["violated"]        = True
                        track["violation_frame"]  = retro["frame"]

                        v = {
                            "track_id":       track_id,
                            "frame":          retro["frame"],
                            "timestamp":      datetime.now().isoformat(),
                            "signal_state":   retro["signal"],
                            "direction":      track["direction"].value,
                            "stop_line_y":    self.stop_line_y,
                            retro["edge_label"]: round(retro["edge_val"], 1),
                            "frames_tracked": track["frames_tracked"],
                            "bbox":           bbox,
                        }
                        self.violations.append(v)
                        events.append(v)
                    continue  # Skip standard check this frame

            direction = track["direction"]

            # Skip standard checks until direction is known
            if direction == Direction.UNKNOWN:
                continue

            # 4. STANDARD REAL-TIME CHECK
            if signal != SignalState.RED:
                track["last_edge_y"] = (top_y if direction == Direction.UP
                                        else bottom_y)
                continue

            # Cooldown check
            if track["violated"]:
                if frame_num - track["violation_frame"] <= COOLDOWN_FRAMES:
                    track["last_edge_y"] = (top_y if direction == Direction.UP
                                            else bottom_y)
                    continue
                else:
                    track["violated"] = False

            # Minimum track frames
            if track["frames_tracked"] < MIN_TRACK_FRAMES:
                track["last_edge_y"] = (top_y if direction == Direction.UP
                                        else bottom_y)
                continue

            last_edge = track["last_edge_y"]
            if last_edge is None:
                track["last_edge_y"] = (top_y if direction == Direction.UP
                                        else bottom_y)
                continue

            violated = False
            ev_edge_label, ev_edge_val = "", 0.0

            if direction == Direction.UP:
                if top_y <= self.stop_line_y < last_edge:
                    violated       = True
                    ev_edge_label  = "top_y"
                    ev_edge_val    = top_y
                track["last_edge_y"] = top_y

            elif direction == Direction.DOWN:
                if bottom_y >= self.stop_line_y > last_edge:
                    violated       = True
                    ev_edge_label  = "bottom_y"
                    ev_edge_val    = bottom_y
                track["last_edge_y"] = bottom_y

            if violated:
                track["violated"]        = True
                track["violation_frame"]  = frame_num

                v = {
                    "track_id":       track_id,
                    "frame":          frame_num,
                    "timestamp":      datetime.now().isoformat(),
                    "signal_state":   signal.value,
                    "direction":      direction.value,
                    "stop_line_y":    self.stop_line_y,
                    ev_edge_label:    round(ev_edge_val, 1),
                    "frames_tracked": track["frames_tracked"],
                    "bbox":           bbox,
                }
                self.violations.append(v)
                events.append(v)

        self._cleanup(frame_num, active_ids)
        return events

    def _cleanup(self, frame_num: int, active_ids: set):
        stale = [tid for tid, t in self.tracks.items()
                 if tid not in active_ids
                 and frame_num - t["last_seen_frame"] > STALE_TRACK_FRAMES]
        for tid in stale:
            del self.tracks[tid]

    def is_violated(self, track_id: int) -> bool:
        t = self.tracks.get(int(track_id))
        return t is not None and t["violated"]

    def get_direction(self, track_id: int) -> Direction:
        t = self.tracks.get(int(track_id))
        return t["direction"] if t else Direction.UNKNOWN

    def get_summary(self) -> dict:
        up_count   = sum(1 for v in self.violations if v["direction"] == "UP")
        down_count = sum(1 for v in self.violations if v["direction"] == "DOWN")
        return {
            "total_violations": len(self.violations),
            "up_violations":    up_count,
            "down_violations":  down_count,
            "violations":       self.violations,
        }



#  OCCLUSION TRACKER
#  Consistent ID reassignment via ghost tracking.


class OcclusionTracker:
    def __init__(self, frame_rate=30):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=frame_rate,
        )
        self.ghost_tracks       = {}
        self.id_map             = {}
        self.next_consistent_id = 1
        self.active_ids         = set()

    def update(self, detections, frame_num):
        tracked     = self.tracker.update_with_detections(detections)
        prev_active = self.active_ids.copy()
        self.active_ids = set()

        if tracked.tracker_id is None:
            return tracked, prev_active - self.active_ids

        new_ids = []
        for i, tid in enumerate(tracked.tracker_id.tolist()):
            x1, y1, x2, y2 = tracked.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if tid in self.id_map:
                cid = self.id_map[tid]
            else:
                cid = self._try_reassign(cx, cy, frame_num)
                if cid is None:
                    cid = self.next_consistent_id
                    self.next_consistent_id += 1
                self.id_map[tid] = cid

            new_ids.append(cid)
            self.active_ids.add(cid)
            self.ghost_tracks[cid] = {
                "last_pos":   (float(cx), float(cy)),
                "lost_frame": int(frame_num),
            }

        self._clean_ghosts(frame_num)
        tracked.tracker_id = new_ids
        return tracked, prev_active - self.active_ids

    def _try_reassign(self, cx, cy, frame_num):
        best, best_dist = None, float("inf")
        for gid, g in self.ghost_tracks.items():
            if frame_num - g["lost_frame"] > 15:
                continue
            d = math.sqrt((float(cx) - g["last_pos"][0]) ** 2 +
                          (float(cy) - g["last_pos"][1]) ** 2)
            if d < 80 and d < best_dist:
                best_dist = d
                best      = gid
        if best is not None:
            del self.ghost_tracks[best]
            return best
        return None

    def _clean_ghosts(self, frame_num):
        stale = [gid for gid, g in self.ghost_tracks.items()
                 if frame_num - g["lost_frame"] > 15]
        for gid in stale:
            del self.ghost_tracks[gid]



#  TVS-9: VIOLATION RULE ENGINE
#  Thin coordinator — owns both sub-detectors, applies cooldown,
#  emits canonical ViolationEvent objects.


class ViolationRuleEngine:
    """
    TVS-9: Combines speed (TVS-7) and red light (TVS-8) detectors.

    Responsibilities:
      1. Receive raw events from both detectors each frame
      2. Apply per-(track, type) cooldown to prevent duplicate events
      3. Emit canonical ViolationEvent objects with full metadata
      4. Maintain a running log for downstream modules (plate crop, OCR, DB)

    The engine owns NO detection logic — it only coordinates and enriches.
    """

    def __init__(self,
                 speed_estimator:    SpeedEstimator,
                 red_light_detector: RedLightDetector,
                 speed_limit_kmh:    float,
                 cooldown_frames:    int = COOLDOWN_FRAMES,
                 evidence_pre:       int = EVIDENCE_PRE_FRAMES,
                 evidence_post:      int = EVIDENCE_POST_FRAMES):

        self.speed_est    = speed_estimator
        self.rl_detector  = red_light_detector
        self.speed_limit  = speed_limit_kmh
        self.cooldown     = cooldown_frames
        self.evidence_pre = evidence_pre
        self.evidence_post = evidence_post

        # Cooldown tracker: {(track_id, ViolationType): last_triggered_frame}
        self._last_triggered: Dict[tuple, int] = {}

        # All emitted events this session
        self.events: List[ViolationEvent] = []
        self._event_counter = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_on_cooldown(self, tid: int, vtype: ViolationType,
                        frame_num: int) -> bool:
        key  = (tid, vtype)
        last = self._last_triggered.get(key, -999999)
        return (frame_num - last) <= self.cooldown

    def _mark_triggered(self, tid: int, vtype: ViolationType,
                        frame_num: int):
        self._last_triggered[(tid, vtype)] = frame_num

    def _make_event_id(self, tid: int, vtype: ViolationType,
                       frame_num: int) -> str:
        self._event_counter += 1
        return f"{tid}_{vtype.value}_{frame_num}_{self._event_counter}"

    def _emit(self,
              tid:        int,
              vtype:      ViolationType,
              frame_num:  int,
              direction:  str,
              signal:     str,
              speed_kmh:  Optional[float],
              bbox:       list) -> ViolationEvent:

        event = ViolationEvent(
            event_id             = self._make_event_id(tid, vtype, frame_num),
            track_id             = tid,
            violation_type       = vtype,
            frame_number         = frame_num,
            timestamp            = datetime.now().isoformat(),
            direction            = direction,
            signal_state         = signal,
            speed_kmh            = speed_kmh,
            speed_limit_kmh      = self.speed_limit if vtype == ViolationType.OVERSPEED else None,
            bbox                 = [round(v, 1) for v in bbox],
            evidence_start_frame = max(0, frame_num - self.evidence_pre),
            evidence_end_frame   = frame_num + self.evidence_post,
        )
        self.events.append(event)
        self._mark_triggered(tid, vtype, frame_num)
        return event

    # ── Main per-frame update ─────────────────────────────────────────────────

    def update(self,
               detections: sv.Detections,
               signal:     SignalState,
               frame_num:  int) -> List[ViolationEvent]:
        """
        Call once per frame with the current detections and signal state.
        Returns list of new ViolationEvents emitted this frame.
        """
        new_events: List[ViolationEvent] = []

        # --- Run sub-detectors ---
        speed_events = self.speed_est.process(detections, frame_num)
        rl_events    = self.rl_detector.process(detections, signal, frame_num)

        # --- Process speed violations ---
        for se in speed_events:
            if not se["violation"]:
                continue
            tid   = se["track_id"]
            vtype = ViolationType.OVERSPEED
            if self._is_on_cooldown(tid, vtype, frame_num):
                continue
            ev = self._emit(
                tid       = tid,
                vtype     = vtype,
                frame_num = se.get("frame_num", frame_num),
                direction = se["direction"],
                signal    = signal.value,
                speed_kmh = se["speed_kmh"],
                bbox      = se.get("bbox", [0, 0, 0, 0]),
            )
            new_events.append(ev)
            print(f"  [RULE ENGINE] Frame {frame_num}: OVERSPEED — "
                  f"Vehicle #{tid} @ {se['speed_kmh']} km/h  "
                  f"[event_id={ev.event_id}]")

        # --- Process red light violations ---
        for rle in rl_events:
            tid   = rle["track_id"]
            vtype = ViolationType.RED_LIGHT
            if self._is_on_cooldown(tid, vtype, frame_num):
                continue

            # Enrich with speed if we happen to have it
            speed = self.speed_est.get_speed(tid)

            ev = self._emit(
                tid       = tid,
                vtype     = vtype,
                frame_num = rle["frame"],
                direction = rle.get("direction", "unknown"),
                signal    = rle.get("signal_state", "RED"),
                speed_kmh = speed,
                bbox      = rle.get("bbox", [0, 0, 0, 0]),
            )
            new_events.append(ev)
            print(f"  [RULE ENGINE] Frame {frame_num}: RED_LIGHT — "
                  f"Vehicle #{tid} direction={rle.get('direction', '?')}  "
                  f"[event_id={ev.event_id}]")

        return new_events

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_events_for_track(self, tid: int) -> List[ViolationEvent]:
        return [e for e in self.events if e.track_id == tid]

    def get_events_by_type(self, vtype: ViolationType) -> List[ViolationEvent]:
        return [e for e in self.events if e.violation_type == vtype]

    def get_summary(self) -> dict:
        overspeed = self.get_events_by_type(ViolationType.OVERSPEED)
        red_light = self.get_events_by_type(ViolationType.RED_LIGHT)
        combined  = [tid for tid in {e.track_id for e in overspeed}
                     if any(e.track_id == tid for e in red_light)]
        return {
            "total_events":    len(self.events),
            "overspeed_count": len(overspeed),
            "red_light_count": len(red_light),
            "combined_count":  len(combined),
            "unique_vehicles": len({e.track_id for e in self.events}),
            "cooldown_frames": self.cooldown,
            "speed_limit_kmh": self.speed_limit,
            "events": [e.to_dict() for e in self.events],
        }



#  DRAWING / HUD


DIR_COLORS = {
    Direction.UP:      (255, 200, 0),     # cyan-ish
    Direction.DOWN:    (0, 165, 255),      # orange
    Direction.UNKNOWN: (160, 160, 160),    # grey
}
DIR_ARROWS = {
    Direction.UP:      "↑",
    Direction.DOWN:    "↓",
    Direction.UNKNOWN: "?",
}


def draw_hud(frame, frame_num: int, engine: ViolationRuleEngine,
             tracked, signal: SignalState, orig_w: int, orig_h: int):
    """Draw combined overlays: speed zone + stop line + traffic light + summary."""

    # Speed zone shading
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, LINE_UPPER_Y), (orig_w, LINE_LOWER_Y),
                  (255, 255, 0), -1)
    frame[:] = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

    # Speed lines
    cv2.line(frame, (0, LINE_UPPER_Y), (orig_w, LINE_UPPER_Y), (0, 0, 255), 2)
    cv2.putText(frame, f"UPPER LINE (Y={LINE_UPPER_Y})",
                (10, LINE_UPPER_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    cv2.line(frame, (0, LINE_LOWER_Y), (orig_w, LINE_LOWER_Y), (0, 255, 0), 2)
    cv2.putText(frame, f"LOWER LINE (Y={LINE_LOWER_Y})",
                (10, LINE_LOWER_Y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # Stop line
    sl_color     = (0, 0, 255) if signal == SignalState.RED else (180, 180, 180)
    sl_thickness = 4           if signal == SignalState.RED else 2
    cv2.line(frame, (0, STOP_LINE_Y), (orig_w, STOP_LINE_Y), sl_color, sl_thickness)
    cv2.putText(frame, f"STOP LINE (Y={STOP_LINE_Y})",
                (10, STOP_LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, sl_color, 2)

    # Direction indicators
    mid_y = (LINE_UPPER_Y + LINE_LOWER_Y) // 2
    cv2.putText(frame, "UP "+chr(8593), (50, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, chr(8595)+" DOWN", (orig_w - 180, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Traffic light widget
    lx, ly = orig_w - 110, 30
    cv2.rectangle(frame, (lx - 15, ly - 10), (lx + 75, ly + 110), (40, 40, 40), -1)
    cv2.rectangle(frame, (lx - 15, ly - 10), (lx + 75, ly + 110), (180, 180, 180), 2)
    for idx, st in enumerate([SignalState.RED, SignalState.YELLOW, SignalState.GREEN]):
        cy_ = ly + 15 + idx * 32
        col = {SignalState.RED: (0,0,255), SignalState.YELLOW: (0,255,255),
               SignalState.GREEN: (0,255,0)}[st]
        if st == signal:
            cv2.circle(frame, (lx + 30, cy_), 13, col, -1)
            cv2.circle(frame, (lx + 30, cy_), 13, (255, 255, 255), 2)
        else:
            cv2.circle(frame, (lx + 30, cy_), 13, (50, 50, 50), -1)
    cv2.putText(frame, signal.value, (lx - 10, ly + 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                {SignalState.GREEN: (0,255,0), SignalState.YELLOW: (0,255,255),
                 SignalState.RED: (0,0,255)}[signal], 2)

    if USE_KEYBOARD:
        cv2.putText(frame, "R/Y/G=signal  Q=quit  P=pause",
                    (10, orig_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Violation flash overlay for recent events
    recent = [e for e in engine.events
              if frame_num - e.frame_number <= EVIDENCE_POST_FRAMES]
    if recent:
        flash_overlay = frame.copy()
        cv2.rectangle(flash_overlay, (0, 0), (orig_w, orig_h), (0, 0, 180), -1)
        frame[:] = cv2.addWeighted(frame, 0.92, flash_overlay, 0.08, 0)

    # Summary box
    summary = engine.get_summary()
    cv2.rectangle(frame, (8, 8), (460, 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (8, 8), (460, 80), (80, 80, 80), 1)
    active = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
    cv2.putText(frame,
                f"Frame:{frame_num} | Active:{active} | Signal:{signal.value}",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(frame,
                f"Violations: {summary['total_events']}  "
                f"(Speed:{summary['overspeed_count']}  "
                f"Red:{summary['red_light_count']}  "
                f"Both:{summary['combined_count']})",
                (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    cv2.putText(frame,
                f"Unique vehicles: {summary['unique_vehicles']}",
                (14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def draw_direction_indicators(frame, tracked, rl_detector: RedLightDetector):
    """Draw a small colored arrow on each vehicle box indicating detected direction."""
    if tracked.tracker_id is None:
        return
    for i, track_id in enumerate(tracked.tracker_id):
        direction = rl_detector.get_direction(track_id)
        color     = DIR_COLORS[direction]
        x1, y1, x2, y2 = tracked.xyxy[i]
        cx = int((x1 + x2) / 2)
        if direction == Direction.UP:
            cv2.arrowedLine(frame, (cx, int(y1) + 20), (cx, int(y1) - 5),
                            color, 2, tipLength=0.4)
        elif direction == Direction.DOWN:
            cv2.arrowedLine(frame, (cx, int(y1) - 5), (cx, int(y1) + 20),
                            color, 2, tipLength=0.4)


def build_labels(tracked, engine: ViolationRuleEngine, model) -> list:
    labels = []
    if tracked.tracker_id is None:
        return labels
    for class_id, tid in zip(tracked.class_id, tracked.tracker_id):
        name      = model.names[class_id]
        speed     = engine.speed_est.get_speed(tid)
        direction = engine.speed_est.get_direction(tid)
        arrow     = "↑" if direction == "up" else "↓" if direction == "down" else "?"
        evs       = engine.get_events_for_track(tid)
        types     = {e.violation_type for e in evs}

        parts = [f"#{tid}", arrow, name]
        if speed is not None:
            parts.append(f"{speed}km/h")
        if ViolationType.OVERSPEED in types:
            parts.append("[!SPEED]")
        if ViolationType.RED_LIGHT in types:
            parts.append("[!RED]")
        labels.append(" ".join(parts))
    return labels


def resize_for_display(frame, target_width=1280):
    h, w = frame.shape[:2]
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)),
                      interpolation=cv2.INTER_AREA)



#  UNIT TESTS


class TestViolationRuleEngine(unittest.TestCase):

    def _make_engine(self, cooldown=COOLDOWN_FRAMES):
        se  = SpeedEstimator(LINE_UPPER_Y, LINE_LOWER_Y,
                             REAL_DISTANCE_METERS, VIDEO_FPS, SPEED_LIMIT_KMH)
        rld = RedLightDetector(STOP_LINE_Y)
        return ViolationRuleEngine(se, rld, SPEED_LIMIT_KMH,
                                   cooldown_frames=cooldown)

    def _make_det(self, tid, x1, y1, x2, y2):
        """Build a minimal sv.Detections mock."""
        import numpy as np
        d = sv.Detections(
            xyxy       = np.array([[x1, y1, x2, y2]], dtype=float),
            confidence = np.array([0.9]),
            class_id   = np.array([2]),
            tracker_id = np.array([tid]),
        )
        return d

    def test_cooldown_prevents_duplicate(self):
        """Same (track, type) within cooldown window must not emit twice."""
        engine = self._make_engine()
        engine._emit(1, ViolationType.OVERSPEED, 100, "up", "GREEN", 80.0,
                     [0, 0, 50, 50])
        # Frame 110 — within cooldown (90 frames)
        on_cd = engine._is_on_cooldown(1, ViolationType.OVERSPEED, 110)
        self.assertTrue(on_cd)
        self.assertEqual(len(engine.events), 1)

    def test_cooldown_expires(self):
        """After cooldown, same track can emit again."""
        engine = self._make_engine()
        engine._emit(1, ViolationType.OVERSPEED, 100, "up", "GREEN", 80.0,
                     [0, 0, 50, 50])
        # Frame 100 + 91 = 191 — past cooldown
        on_cd = engine._is_on_cooldown(1, ViolationType.OVERSPEED, 191)
        self.assertFalse(on_cd)

    def test_different_types_independent_cooldown(self):
        """OVERSPEED and RED_LIGHT cooldowns are independent per track."""
        engine = self._make_engine()
        engine._emit(1, ViolationType.OVERSPEED, 100, "up", "GREEN", 80.0,
                     [0, 0, 50, 50])
        engine._emit(1, ViolationType.RED_LIGHT, 100, "up", "RED", None,
                     [0, 0, 50, 50])
        self.assertEqual(len(engine.events), 2)
        # Both are on cooldown individually
        self.assertTrue(engine._is_on_cooldown(1, ViolationType.OVERSPEED, 150))
        self.assertTrue(engine._is_on_cooldown(1, ViolationType.RED_LIGHT, 150))

    def test_different_tracks_independent(self):
        """Cooldown on track 1 must not affect track 2."""
        engine = self._make_engine()
        engine._emit(1, ViolationType.OVERSPEED, 100, "up", "GREEN", 80.0,
                     [0, 0, 50, 50])
        on_cd = engine._is_on_cooldown(2, ViolationType.OVERSPEED, 110)
        self.assertFalse(on_cd)

    def test_event_fields(self):
        """ViolationEvent must carry all required fields for downstream modules."""
        engine = self._make_engine()
        ev = engine._emit(5, ViolationType.RED_LIGHT, 200, "down", "RED",
                          None, [10, 20, 60, 80])
        self.assertEqual(ev.track_id, 5)
        self.assertEqual(ev.violation_type, ViolationType.RED_LIGHT)
        self.assertEqual(ev.frame_number, 200)
        self.assertEqual(ev.evidence_start_frame, 200 - EVIDENCE_PRE_FRAMES)
        self.assertEqual(ev.evidence_end_frame, 200 + EVIDENCE_POST_FRAMES)
        self.assertIsNone(ev.speed_kmh)
        self.assertEqual(ev.plate_number, "")  # unfilled until TVS-10

    def test_event_id_unique(self):
        """Every emitted event must have a unique event_id."""
        engine = self._make_engine()
        ids = [
            engine._emit(i, ViolationType.OVERSPEED, 100 + i, "up", "G",
                         80.0, [0, 0, 1, 1]).event_id
            for i in range(10)
        ]
        self.assertEqual(len(ids), len(set(ids)))

    def test_summary_counts(self):
        """Summary must correctly count by type and unique vehicles."""
        engine = self._make_engine()
        engine._emit(1, ViolationType.OVERSPEED, 100, "up", "G", 80.0,
                     [0, 0, 1, 1])
        engine._emit(2, ViolationType.RED_LIGHT, 200, "up", "R", None,
                     [0, 0, 1, 1])
        engine._emit(3, ViolationType.OVERSPEED, 300, "up", "G", 90.0,
                     [0, 0, 1, 1])
        engine._emit(3, ViolationType.RED_LIGHT, 300, "up", "R", 90.0,
                     [0, 0, 1, 1])
        s = engine.get_summary()
        self.assertEqual(s["total_events"],    4)
        self.assertEqual(s["overspeed_count"], 2)
        self.assertEqual(s["red_light_count"], 2)
        self.assertEqual(s["combined_count"],  1)   # vehicle #3
        self.assertEqual(s["unique_vehicles"], 3)

    def test_to_dict_serializable(self):
        """ViolationEvent.to_dict() must produce JSON-serializable output."""
        engine = self._make_engine()
        ev = engine._emit(1, ViolationType.OVERSPEED, 100, "up", "G", 80.0,
                          [0, 0, 50, 80])
        d = ev.to_dict()
        json.dumps(d)  # must not raise


class TestBangladeshPlateRecognizer(unittest.TestCase):

    def test_normalizes_bengali_digits(self):
        value = BangladeshPlateRecognizer.normalize_text("ঢাকা মেট্রো-গ ১২-৩৪৫৬")
        self.assertEqual(value, "ঢাকা মেট্রো-গ 12-3456")

    def test_removes_unsafe_punctuation(self):
        value = BangladeshPlateRecognizer.normalize_text(" ঢাকা@@ মেট্রো গ: ১২৩৪ ")
        self.assertEqual(value, "ঢাকা মেট্রো গ 1234")

    def test_rejects_short_ocr_fragment(self):
        self.assertFalse(BangladeshPlateRecognizer.is_plausible_plate("TEU"))
        self.assertTrue(BangladeshPlateRecognizer.is_plausible_plate("ঢাকা গ 1234"))

    def test_rejects_non_bangladesh_plate(self):
        self.assertFalse(BangladeshPlateRecognizer.is_plausible_plate("WB 04 E3439"))

    def test_invalid_easyocr_result_does_not_skip_tesseract(self):
        import numpy as np
        recognizer = BangladeshPlateRecognizer(lazy=True)
        crop = np.full((40, 120, 3), 180, dtype=np.uint8)
        recognizer._run_easyocr = lambda _images: OCRCandidate("=", 0.95, "easyocr")
        recognizer._run_tesseract = lambda _images: OCRCandidate(
            "ঢাকা গ ১২৩৪", 0.60, "tesseract"
        )
        result = recognizer.recognize(crop)
        self.assertEqual(result.engine, "tesseract")

    def test_invalid_bbox_returns_no_crop(self):
        import numpy as np
        recognizer = BangladeshPlateRecognizer(lazy=True)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(recognizer.locate_plate(frame, [0, 0, 0, 0]))

    def test_estimates_red_vehicle_color(self):
        import numpy as np
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[10:90, 20:140] = (0, 0, 220)
        color, confidence = BangladeshPlateRecognizer.estimate_vehicle_color(
            frame, [20, 10, 140, 90]
        )
        self.assertEqual(color, "RED")
        self.assertGreater(confidence, 0.9)

    def test_visual_score_rejects_blank_bumper_crop(self):
        import numpy as np
        blank = np.full((40, 160, 3), 120, dtype=np.uint8)
        self.assertLess(BangladeshPlateRecognizer._plate_visual_score(blank), 0)

    def test_multiframe_selection_prefers_plausible_plate_text(self):
        import numpy as np
        recognizer = BangladeshPlateRecognizer(lazy=True)
        noisy_crop = np.full((40, 140, 3), 30, dtype=np.uint8)
        valid_crop = np.full((70, 140, 3), 220, dtype=np.uint8)
        frame_a = np.zeros((100, 160, 3), dtype=np.uint8)
        frame_b = np.ones((100, 160, 3), dtype=np.uint8)

        def fake_locate(frame, _bbox):
            return (5.0, noisy_crop) if frame[0, 0, 0] == 0 else (3.0, valid_crop)

        def fake_recognize(crop):
            if crop.shape[0] == noisy_crop.shape[0]:
                return OCRCandidate("=", 0.95, "tesseract")
            return OCRCandidate("ঢাকা গ ১২৩৪", 0.60, "easyocr")

        recognizer.locate_plate_candidate = fake_locate
        recognizer.recognize = fake_recognize
        event = ViolationEvent(
            event_id="test_event", track_id=1,
            violation_type=ViolationType.RED_LIGHT, frame_number=10,
            timestamp="2026-07-03T00:00:00", direction="DOWN",
            signal_state="RED", speed_kmh=None, speed_limit_kmh=None,
            bbox=[0, 0, 160, 100], evidence_start_frame=1,
            evidence_end_frame=20,
        )

        with mock.patch.object(cv2, "imwrite", return_value=True):
            recognizer.process_event_candidates(
                [(frame_a, event.bbox), (frame_b, event.bbox)], event
            )

        self.assertEqual(event.plate_number, "ঢাকা গ 1234")
        self.assertEqual(event.ocr_engine, "easyocr")



#  MAIN


def main():
    import sys
    if "--test" in sys.argv:
        print("Running TVS-9 + Epic 4 unit tests...")
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestViolationRuleEngine))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBangladeshPlateRecognizer))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1

    if "--ocr-image" in sys.argv:
        try:
            image_path = Path(sys.argv[sys.argv.index("--ocr-image") + 1])
        except (ValueError, IndexError):
            print("Usage: python plate_recognition_pipeline.py --ocr-image IMAGE_PATH")
            return 2
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"ERROR: Cannot open image: {image_path}")
            return 2
        recognizer = BangladeshPlateRecognizer(lazy=False)
        result = recognizer.recognize(image)
        cleaned = recognizer.normalize_text(result.text)
        plate = (
            cleaned
            if result.confidence >= OCR_CONFIDENCE_MIN
            and recognizer.is_plausible_plate(cleaned)
            else "UNREADABLE"
        )
        print(json.dumps({
            "plate_number": plate,
            "ocr_raw_text": result.text,
            "ocr_confidence": round(result.confidence, 4),
            "ocr_engine": result.engine,
        }, ensure_ascii=False, indent=2))
        return 0

    video_path = VIDEO_PATH
    if "--video" in sys.argv:
        try:
            video_path = sys.argv[sys.argv.index("--video") + 1]
        except (ValueError, IndexError):
            print("Usage: python plate_recognition_pipeline.py --video VIDEO_PATH")
            return 2

    print("=" * 60)
    print("EPIC 4: Bangladesh Plate Recognition Pipeline")
    print("  Speed (TVS-7) + Red Light (TVS-8) + Rule Engine (TVS-9)")
    print("  Plate Crop (TVS-10) + OCR (TVS-11) + Failure Handling (TVS-12)")
    print("=" * 60)
    print(f"Speed zone    : Y={LINE_UPPER_Y} (upper) to Y={LINE_LOWER_Y} (lower)")
    print(f"Real distance : {REAL_DISTANCE_METERS} m")
    print(f"Speed limit   : {SPEED_LIMIT_KMH} km/h")
    print(f"Stop line     : Y={STOP_LINE_Y}")
    print(f"Cooldown      : {COOLDOWN_FRAMES} frames")
    print(f"Evidence clip : -{EVIDENCE_PRE_FRAMES} / +{EVIDENCE_POST_FRAMES} frames")
    print(f"Signal mode   : {'KEYBOARD (R/Y/G)' if USE_KEYBOARD else 'AUTO CYCLE'}")
    print(f"FPS           : {VIDEO_FPS}")
    print("-" * 60)

    model   = YOLO(MODEL_PATH)
    tracker = OcclusionTracker(frame_rate=VIDEO_FPS)
    light   = TrafficLight()

    speed_est = SpeedEstimator(
        LINE_UPPER_Y, LINE_LOWER_Y,
        REAL_DISTANCE_METERS, VIDEO_FPS, SPEED_LIMIT_KMH
    )
    rl_det = RedLightDetector(STOP_LINE_Y)
    engine = ViolationRuleEngine(speed_est, rl_det, SPEED_LIMIT_KMH)
    plate_recognizer = BangladeshPlateRecognizer(lazy=True)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from epic5 import ViolationDBWriter
    db_writer = ViolationDBWriter()
    print(f"Database      : {'CONNECTED' if db_writer.ping() else 'OFFLINE (queue enabled)'}")
    flushed = db_writer.flush_pending()
    if flushed:
        print(f"  [DB] Flushed {flushed} queued event(s)")

    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.55)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video:", video_path)
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video         : {video_path} ({orig_w}x{orig_h})")
    print("Controls      : Q=quit  P=pause  R=Red  Y=Yellow  G=Green")
    print("=" * 60)

    frame_count = 0
    paused      = False
    frame_history = deque(maxlen=EVIDENCE_PRE_FRAMES + 1)
    pending_ocr: Dict[str, ViolationEvent] = {}
    last_ocr_attempt: Dict[str, int] = {}

    while True:
        # Key read FIRST — used for signal update this same frame
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print("  [PAUSED]" if paused else "  [RESUMED]")

        if USE_KEYBOARD:
            if key in (ord('r'), ord('y'), ord('g')):
                light.set_state(key)
        else:
            if not paused:
                light.auto_update(frame_count)

        if paused:
            continue

        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Detection & tracking
        results    = model(frame, classes=VEHICLE_CLASSES,
                           conf=CONFIDENCE, iou=IOU, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        tracked, _ = tracker.update(detections, frame_count)

        # Retain the promised pre-event window. Earlier frames are often sharper
        # for vehicles moving away from the camera.
        history_boxes = {}
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                history_boxes[int(tid)] = tracked.xyxy[i].tolist()
        frame_history.append((frame_count, frame.copy(), history_boxes))

        # ── Single call to the rule engine ────────────────────────────────
        new_events = engine.update(tracked, light.state, frame_count)
        for event in new_events:
            pending_ocr[event.event_id] = event
            candidates = []
            for old_num, old_frame, old_boxes in frame_history:
                old_bbox = old_boxes.get(event.track_id)
                if old_bbox is None:
                    continue
                candidates.append((old_frame, old_bbox))
            if candidates:
                # Plate localization and OCR quality—not whole-vehicle size—now
                # decide which evidence frame is retained.
                plate_recognizer.process_event_candidates(candidates, event)
                last_ocr_attempt[event.event_id] = frame_count
            else:
                plate_recognizer.process_event(frame, event)
                last_ocr_attempt[event.event_id] = frame_count
            db_writer.write_event(event)

        # Retry unreadable plates while the post-event evidence window remains.
        for event_id, event in list(pending_ocr.items()):
            if event.plate_number != "UNREADABLE":
                pending_ocr.pop(event_id, None)
                continue
            if frame_count >= event.evidence_end_frame:
                pending_ocr.pop(event_id, None)
                continue
            if frame_count - last_ocr_attempt.get(event_id, -9999) < OCR_RETRY_FRAMES:
                continue
            if tracked.tracker_id is None:
                continue
            for i, tid in enumerate(tracked.tracker_id):
                if int(tid) != event.track_id:
                    continue
                plate_recognizer.process_event(frame, event, tracked.xyxy[i].tolist())
                last_ocr_attempt[event_id] = frame_count
                db_writer.write_event(event)
                break

        # Draw scene
        annotated = frame.copy()
        draw_hud(annotated, frame_count, engine, tracked,
                 light.state, orig_w, orig_h)
        draw_direction_indicators(annotated, tracked, engine.rl_detector)

        labels    = build_labels(tracked, engine, model)
        annotated = box_annotator.annotate(scene=annotated, detections=tracked)
        annotated = label_annotator.annotate(scene=annotated,
                                              detections=tracked,
                                              labels=labels)

        display = resize_for_display(annotated, DISPLAY_WIDTH)
        cv2.imshow("Epic 4 Bangladesh Plate Recognition", display)

    cap.release()
    cv2.destroyAllWindows()

    # ── Final Report ──────────────────────────────────────────────────────
    summary = engine.get_summary()
    speed_summary = engine.speed_est.get_summary()
    rl_summary    = engine.rl_detector.get_summary()

    print("\n" + "=" * 60)
    print("EPIC 4 BANGLADESH PLATE RECOGNITION — FINAL REPORT")
    print("=" * 60)
    print(f"Total events      : {summary['total_events']}")
    print(f"  Overspeed       : {summary['overspeed_count']}")
    print(f"  Red light       : {summary['red_light_count']}")
    print(f"  Both types      : {summary['combined_count']}")
    print(f"Unique vehicles   : {summary['unique_vehicles']}")

    print(f"\nSpeed measurements: {speed_summary['total_valid']} "
          f"(UP: {speed_summary['up_count']}, DOWN: {speed_summary['down_count']})")
    print(f"  Discarded       : {speed_summary['discarded']}")
    print(f"  Average speed   : {speed_summary['average_speed']} km/h")
    print(f"  Calibration     : {speed_summary['pixels_per_meter']} px/m")

    print(f"\nRed light hits    : {rl_summary['total_violations']} "
          f"(UP: {rl_summary['up_violations']}, DOWN: {rl_summary['down_violations']})")

    if summary["events"]:
        print("\nEvent log:")
        for e in summary["events"]:
            speed_str = f"{e['speed_kmh']} km/h" if e["speed_kmh"] else "N/A"
            print(f"  [{e['event_id']}]  "
                  f"Frame {e['frame_number']:>5}  "
                  f"Vehicle #{e['track_id']:>3}  "
                  f"{e['violation_type']:<10}  "
                  f"speed={speed_str:<12}  "
                  f"dir={e['direction']:<5}  "
                  f"signal={e['signal_state']}")
            print(f"           evidence frames: "
                  f"{e['evidence_start_frame']} -> {e['evidence_end_frame']}")
            print(f"           plate={e['plate_number']}  "
                  f"ocr={e['ocr_engine']} ({e['ocr_confidence']:.1%})")
            print(f"           color={e['vehicle_color']} "
                  f"({e['color_confidence']:.1%})")

    # Save combined JSON report
    combined_report = {
        "rule_engine": summary,
        "speed_estimation": speed_summary,
        "red_light_detection": rl_summary,
    }
    out = "violation_events.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")
    print("Run with --test flag to execute unit tests.")
    print("=" * 60)


if __name__ == "__main__":
    main()

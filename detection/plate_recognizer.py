"""
TVS-10/11/12: Bangladesh license-plate localization and OCR.

Localizes plates with the supplied YOLO plate-detector model (falling back
to OpenCV contour localization when the model finds nothing plausible),
runs Bengali/English OCR (EasyOCR primary, Tesseract fallback), scores
confidence, and stores evidence crops for the dashboard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import cv2
import easyocr
import pytesseract
from ultralytics import YOLO

from .config import (
    PROJECT_ROOT, EVIDENCE_DIR, EASYOCR_MODEL_DIR, TESSDATA_DIR,
    PLATE_MODEL_PATH, OCR_CONFIDENCE_MIN, PLATE_DETECT_CONF,
    PLATE_MIN_AREA_RATIO, PLATE_MAX_AREA_RATIO, PLATE_OCR_TOP_K,
    PLATE_MIN_RAW_WIDTH, PLATE_MIN_RAW_HEIGHT,
)
from .models import ViolationEvent


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



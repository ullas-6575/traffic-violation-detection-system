"""
plate_module/pipeline.py
TVS-10: Plate Recognition Pipeline — the only class the rest of the
project needs to talk to.

Combines:
  - PlateLocator   (detector.py)  -> finds the plate rectangle
  - PlateOCRReader (ocr_reader.py) -> reads text from that rectangle

Responsibilities:
  1. Crop the vehicle out of the full frame using the violation bbox
  2. Locate the plate inside that crop
  3. Run OCR on the plate region
  4. Save an evidence image (vehicle crop with plate box drawn)
  5. Cache per-track_id results so the same vehicle isn't OCR'd on
     every single frame — once a confident reading is found it's
     reused; low-confidence/failed reads retry after a cooldown
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from . import config as cfg
from .detector import PlateLocator, OpenCVPlateLocator, YOLOPlateLocator
from .ocr_reader import PlateOCRReader, OCRResult


@dataclass
class PlateRecognitionResult:
    plate_number: str        # "" if not (yet) read successfully
    confidence: float
    image_path: str          # "" if no crop was saved
    raw_text: str = ""
    ocr_engine: str = "none"
    plate_crop_path: str = ""


class PlateRecognitionPipeline:
    """
    Single entry point: recognize(frame, bbox, track_id, frame_num) -> PlateRecognitionResult

    Pass a custom `locator` (e.g. YOLOPlateLocator once you have weights)
    to swap detection strategy without touching this class or the
    violation_rule_engine.
    """

    def __init__(self,
                 locator: Optional[PlateLocator] = None,
                 reader: Optional[PlateOCRReader] = None,
                 crops_dir: str = cfg.PLATE_CROPS_DIR,
                 retry_cooldown_frames: int = cfg.RETRY_COOLDOWN_FRAMES):
        self.locator = locator or OpenCVPlateLocator()
        self.reader = reader or PlateOCRReader()
        self.crops_dir = crops_dir
        self.retry_cooldown_frames = retry_cooldown_frames
        os.makedirs(self.crops_dir, exist_ok=True)

        # track_id -> PlateRecognitionResult, only stored once confident
        self._confident_cache: Dict[int, PlateRecognitionResult] = {}
        # track_id -> last frame_num we attempted OCR on (for retry cooldown)
        self._last_attempt_frame: Dict[int, int] = {}

    # ── Cache management ───────────────────────────────────────────────────

    def forget_track(self, track_id: int):
        """Remove a track from caches so a reused ID doesn't carry stale data."""
        self._confident_cache.pop(track_id, None)
        self._last_attempt_frame.pop(track_id, None)

    def forget_all(self):
        self._confident_cache.clear()
        self._last_attempt_frame.clear()

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _crop_vehicle(frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def preview_plate_box(self, frame: np.ndarray, bbox: list) -> Optional[Tuple[int, int, int, int]]:
        """
        Lightweight preview: finds the plate rectangle and returns it in
        FRAME coordinates (not crop coordinates). No OCR, no saving, no caching.
        """
        vehicle_crop = self._crop_vehicle(frame, bbox)
        if vehicle_crop is None:
            return None

        plate_box = self.locator.locate(vehicle_crop)
        if plate_box is None:
            return None

        px1, py1, px2, py2 = plate_box
        vx1, vy1, _, _ = [int(v) for v in bbox]
        return (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)

    def _save_evidence(self, vehicle_crop: np.ndarray,
                        plate_box: Optional[Tuple[int, int, int, int]],
                        track_id: int, frame_num: int) -> str:
        annotated = vehicle_crop.copy()
        if plate_box is not None:
            px1, py1, px2, py2 = plate_box
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)

        base = f"track{track_id}_frame{frame_num}"
        filename = f"{base}.jpg"
        path = os.path.join(self.crops_dir, filename)

        # Avoid overwriting previous crops for the same track/frame
        counter = 1
        while os.path.exists(path):
            filename = f"{base}_{counter}.jpg"
            path = os.path.join(self.crops_dir, filename)
            counter += 1

        cv2.imwrite(path, annotated)
        return path

    def _save_plate_crop(self, plate_crop: np.ndarray, track_id: int,
                         frame_num: int) -> str:
        if plate_crop is None or plate_crop.size == 0:
            return ""
        path = os.path.join(self.crops_dir, f"track{track_id}_frame{frame_num}_plate.jpg")
        cv2.imwrite(path, plate_crop)
        return path

    # ── Main entry point ─────────────────────────────────────────────────

    def recognize(self, frame: np.ndarray, bbox: list,
                   track_id: int, frame_num: int) -> PlateRecognitionResult:
        """
        frame:     full BGR video frame
        bbox:      [x1, y1, x2, y2] of the vehicle, in frame coordinates
                   (this is the same bbox already stored on ViolationEvent)
        track_id:  vehicle's track id
        frame_num: current frame number (used for evidence filenames and
                   retry cooldown bookkeeping)
        """
        # Already have a confident reading for this vehicle — reuse it.
        if track_id in self._confident_cache:
            return self._confident_cache[track_id]

        # Respect retry cooldown so we don't run OCR every single frame
        # for a vehicle whose plate keeps failing to read.
        last_attempt = self._last_attempt_frame.get(track_id, -10_000)
        if frame_num - last_attempt < self.retry_cooldown_frames:
            return PlateRecognitionResult(plate_number="", confidence=0.0, image_path="")

        self._last_attempt_frame[track_id] = frame_num

        vehicle_crop = self._crop_vehicle(frame, bbox)
        if vehicle_crop is None:
            return PlateRecognitionResult(plate_number="", confidence=0.0, image_path="")

        plate_box = self.locator.locate(vehicle_crop)

        ocr_result: OCRResult
        if plate_box is not None:
            px1, py1, px2, py2 = plate_box
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            ocr_result = self.reader.read(plate_crop)
        else:
            # Fall back to OCR-ing the whole vehicle crop's lower half —
            # better than nothing if localization fails outright.
            h = vehicle_crop.shape[0]
            fallback_crop = vehicle_crop[int(h * 0.5):, :]
            ocr_result = self.reader.read(fallback_crop)
            plate_crop = fallback_crop

        image_path = self._save_evidence(vehicle_crop, plate_box, track_id, frame_num)
        plate_crop_path = self._save_plate_crop(plate_crop, track_id, frame_num)

        result = PlateRecognitionResult(
            plate_number=ocr_result.text,
            confidence=ocr_result.confidence,
            image_path=image_path,
            raw_text=ocr_result.raw_text,
            ocr_engine=ocr_result.engine,
            plate_crop_path=plate_crop_path,
        )

        if ocr_result.text:
            # Only cache (and stop retrying) once we have a real reading.
            self._confident_cache[track_id] = result

        return result

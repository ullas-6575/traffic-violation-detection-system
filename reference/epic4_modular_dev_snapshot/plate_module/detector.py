"""
plate_module/detector.py
TVS-10: Plate localization using pre-trained YOLO license plate detector.

Uses:
  1. YOLO plate detector (license_plate_detector.pt) — trained on real plates
  2. DynamicPlateLocator fallback — if YOLO model missing or fails
"""

from __future__ import annotations

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import os

from . import config as cfg


class PlateLocator(ABC):
    """Abstract interface: locate a plate rectangle inside a vehicle crop."""

    @abstractmethod
    def locate(self, vehicle_crop: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        raise NotImplementedError


class YOLOPlateDetector(PlateLocator):
    """
    Pre-trained YOLOv8 license plate detector.
    Model: license_plate_detector.pt (trained on Roboflow license plate dataset)
    """

    def __init__(self, model_path: str = "models/license_plate_detector.pt",
                 conf: float = 0.3):
        self.conf = conf
        self.model = None
        
        try:
            from ultralytics import YOLO
            
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"  [YOLO Plate] Loaded plate detector: {model_path}")
            else:
                print(f"  [YOLO Plate] Model not found at {model_path}, will use fallback")
                
        except Exception as e:
            print(f"  [YOLO Plate] Failed to load: {e}")

    def locate(self, vehicle_crop: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self.model is None or vehicle_crop is None or vehicle_crop.size == 0:
            return None

        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 40:
            return None

        # Run YOLO plate detection on the vehicle crop
        results = self.model(vehicle_crop, verbose=False, conf=self.conf)

        if not results or len(results) == 0:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        # Get the best plate detection (highest confidence)
        best = boxes[boxes.conf.argmax()]
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy()

        # Add small padding
        pad_x = int((x2 - x1) * 0.03)
        pad_y = int((y2 - y1) * 0.03)

        x1 = max(0, int(x1) - pad_x)
        y1 = max(0, int(y1) - pad_y)
        x2 = min(w, int(x2) + pad_x)
        y2 = min(h, int(y2) + pad_y)

        return (x1, y1, x2, y2)


class DynamicPlateLocator(PlateLocator):
    """
    Multi-strategy dynamic plate locator (fallback when YOLO plate model unavailable).
    """

    def __init__(self,
                 min_area_px: int = cfg.PLATE_MIN_AREA_PX,
                 max_area_ratio: float = cfg.PLATE_MAX_AREA_RATIO,
                 aspect_min: float = cfg.PLATE_ASPECT_MIN,
                 aspect_max: float = cfg.PLATE_ASPECT_MAX):
        self.min_area_px = min_area_px
        self.max_area_ratio = max_area_ratio
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max

    def locate(self, vehicle_crop: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 40:
            return None

        candidates: List[Tuple[Tuple[int, int, int, int], float]] = []

        for strategy_name, boxes in [
            ("text_edges", self._strategy_text_edges(vehicle_crop)),
            ("bright_blob", self._strategy_bright_blob(vehicle_crop)),
            ("dark_on_light", self._strategy_dark_on_light(vehicle_crop)),
        ]:
            for box in boxes:
                score = self._score_candidate(vehicle_crop, box, strategy_name)
                if score > 0:
                    candidates.append((box, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _score_candidate(self, vehicle_crop, box, strategy):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        h, w = vehicle_crop.shape[:2]

        if bh == 0 or bw == 0:
            return -1

        aspect = bw / bh
        if not (self.aspect_min <= aspect <= self.aspect_max):
            return -1

        area = bw * bh
        crop_area = h * w
        if area < self.min_area_px or area > crop_area * self.max_area_ratio:
            return -1

        candidate = vehicle_crop[y1:y2, x1:x2]
        if candidate.size == 0:
            return -1

        gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)

        score = 0.0

        aspect_score = 1.0 - min(abs(aspect - 4.5) / 3.0, 1.0)
        score += aspect_score * 25.0

        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.countNonZero(edges) / (bw * bh)
        if 0.03 <= edge_density <= 0.35:
            edge_score = 1.0 - abs(edge_density - 0.15) / 0.15
            score += max(edge_score, 0) * 30.0

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h_proj = np.sum(thresh, axis=1) / 255.0
        if len(h_proj) > 3:
            h_var = np.var(h_proj)
            char_score = min(h_var / 5000.0, 1.0)
            score += char_score * 20.0

        y_center = (y1 + y2) / 2
        y_ratio = y_center / h
        if 0.5 <= y_ratio <= 0.95:
            score += 15.0
        elif 0.3 <= y_ratio < 0.5:
            score += 5.0

        if strategy == "text_edges":
            score += 10.0

        return score

    def _strategy_text_edges(self, vehicle_crop):
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, h_kernel)

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, v_kernel)

        return self._contours_to_boxes(closed)

    def _strategy_bright_blob(self, vehicle_crop):
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        bright = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, 5)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

        return self._contours_to_boxes(closed)

    def _strategy_dark_on_light(self, vehicle_crop):
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, h_kernel)

        return self._contours_to_boxes(dark)

    def _contours_to_boxes(self, binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw > 10 and bh > 5:
                boxes.append((x, y, x + bw, y + bh))
        return boxes


class HybridPlateLocator(PlateLocator):
    """
    Best of both worlds:
      1. Try YOLO plate detector first (trained on real plates, most accurate)
      2. If YOLO fails or model missing, use DynamicPlateLocator
    """

    def __init__(self, model_path: str = "models/license_plate_detector.pt"):
        self.yolo = YOLOPlateDetector(model_path=model_path)
        self.dynamic = DynamicPlateLocator()

    @staticmethod
    def _is_plausible(vehicle_crop, box) -> bool:
        """Reject confident YOLO boxes that contain blank bumper/road regions."""
        x1, y1, x2, y2 = box
        candidate = vehicle_crop[y1:y2, x1:x2]
        if candidate.size == 0:
            return False
        gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        edge_density = cv2.countNonZero(cv2.Canny(gray, 50, 150)) / gray.size
        contrast = float(gray.std())
        return contrast >= 12.0 and 0.015 <= edge_density <= 0.55

    def locate(self, vehicle_crop: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        # Try YOLO plate detector first
        if self.yolo.model is not None:
            box = self.yolo.locate(vehicle_crop)
            if box is not None and self._is_plausible(vehicle_crop, box):
                return box

        # Fall back to dynamic locator
        return self.dynamic.locate(vehicle_crop)


class OpenCVPlateLocator(PlateLocator):
    """
    Backward-compatible alias. Now uses HybridPlateLocator with YOLO.
    """

    def __init__(self, **kwargs):
        self._hybrid = HybridPlateLocator()

    def locate(self, vehicle_crop: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        return self._hybrid.locate(vehicle_crop)


class YOLOPlateLocator(PlateLocator):
    """
    STUB for custom YOLO plate-detector weights.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path
        if weights_path is not None:
            raise NotImplementedError(
                "YOLOPlateLocator is a stub — load your model in __init__ "
                "and implement locate() before passing weights_path."
            )

    def locate(self, vehicle_crop: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        raise NotImplementedError("Implement YOLOPlateLocator before using it.")

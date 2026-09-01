"""
plate_module/ocr_reader.py
TVS-10: Reads text out of a plate crop using Tesseract OCR.
"""

from __future__ import annotations

import re
import cv2
import numpy as np
try:
    import pytesseract
except ImportError:
    pytesseract = None
from dataclasses import dataclass
from typing import Optional

from . import config as cfg

if pytesseract is not None and cfg.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_CMD


@dataclass
class OCRResult:
    text: str
    confidence: float
    raw_text: str
    engine: str = "none"


class PlateOCRReader:
    """Tesseract-backed OCR for license plate crops."""

    def __init__(self,
                 lang: str = cfg.OCR_LANG,
                 psm: int = cfg.OCR_PSM,
                 whitelist: str = cfg.OCR_WHITELIST,
                 min_confidence: float = cfg.OCR_MIN_CONFIDENCE,
                 min_text_len: int = cfg.OCR_MIN_TEXT_LEN,
                 upscale_factor: float = cfg.OCR_UPSCALE_FACTOR):
        self.lang = lang
        self.min_confidence = min_confidence
        self.min_text_len = min_text_len
        self.upscale_factor = upscale_factor
        self.tess_config = (
            f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
        )
        self.easyocr_reader = None
        if cfg.OCR_USE_EASYOCR:
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(
                    cfg.OCR_EASYOCR_LANGUAGES, gpu=False,
                    download_enabled=cfg.OCR_DOWNLOAD_MODELS
                )
            except Exception as exc:
                print(f"  [OCR] EasyOCR unavailable: {exc}")

    def _preprocess(self, plate_crop: np.ndarray) -> np.ndarray:
        # Defensive: handle already-grayscale input
        if len(plate_crop.shape) == 3 and plate_crop.shape[2] == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop

        h, w = gray.shape[:2]
        gray = cv2.resize(
            gray,
            (int(w * self.upscale_factor), int(h * self.upscale_factor)),
            interpolation=cv2.INTER_CUBIC,
        )

        gray = cv2.bilateralFilter(gray, 9, 17, 17)

        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresh

    @staticmethod
    def _clean(raw_text: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text)
        return cleaned.upper()

    def read(self, plate_crop: np.ndarray) -> OCRResult:
        if plate_crop is None or plate_crop.size == 0:
            return OCRResult(text="", confidence=0.0, raw_text="", engine="none")

        processed = self._preprocess(plate_crop)
        easy_result = OCRResult(text="", confidence=0.0, raw_text="", engine="none")

        # TVS-11: EasyOCR is primary. Its confidence is normalized to 0..100.
        if self.easyocr_reader is not None:
            try:
                # EasyOCR's detector performs better on contrast-enhanced grayscale
                # than on hard-thresholded text, especially for tiny video plates.
                if len(plate_crop.shape) == 3:
                    easy_input = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                else:
                    easy_input = plate_crop
                easy_input = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(easy_input)
                easy_input = cv2.resize(
                    easy_input, None, fx=self.upscale_factor, fy=self.upscale_factor,
                    interpolation=cv2.INTER_CUBIC,
                )
                candidates = self.easyocr_reader.readtext(
                    easy_input, detail=1, allowlist=cfg.OCR_WHITELIST,
                    text_threshold=0.35, low_text=0.25, link_threshold=0.25,
                )
                if candidates:
                    raw = " ".join(str(item[1]) for item in candidates)
                    cleaned = self._clean(raw)
                    confidence = max(float(item[2]) for item in candidates) * 100.0
                    easy_result = OCRResult("", round(confidence, 1), raw, "easyocr")
                    if len(cleaned) >= self.min_text_len and confidence >= self.min_confidence:
                        return OCRResult(cleaned, round(confidence, 1), raw, "easyocr")
                else:
                    easy_result = OCRResult("", 0.0, "", "easyocr")
            except Exception as exc:
                print(f"  [OCR] EasyOCR read failed, using Tesseract: {exc}")

        # Tesseract fallback. Missing binaries/packages must not crash a violation run.
        try:
            if pytesseract is None:
                raise RuntimeError("pytesseract package is not installed")
            data = pytesseract.image_to_data(
                processed, lang=self.lang, config=self.tess_config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            print(f"  [OCR] Tesseract unavailable/read failed: {exc}")
            return easy_result

        words, confidences = [], []
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            conf = float(conf)
            if not text or conf < 0:
                continue
            words.append(text)
            confidences.append(conf)

        raw_text = " ".join(words)
        cleaned = self._clean(raw_text)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

        if len(cleaned) < self.min_text_len or mean_conf < self.min_confidence:
            if easy_result.confidence > mean_conf:
                return easy_result
            return OCRResult(text="", confidence=round(mean_conf, 1), raw_text=raw_text,
                             engine="tesseract")

        return OCRResult(text=cleaned, confidence=round(mean_conf, 1), raw_text=raw_text,
                         engine="tesseract")

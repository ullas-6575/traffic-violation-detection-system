"""
plate_module/test_plate_module.py
TVS-10: Unit tests for plate localization, OCR cleanup, and the
recognition pipeline's caching/retry logic.

Run with:
    python -m unittest plate_module.test_plate_module -v
"""

import os
import shutil
import unittest
import numpy as np
import cv2

from .detector import OpenCVPlateLocator
from .ocr_reader import PlateOCRReader
from .pipeline import PlateRecognitionPipeline


class TestOpenCVPlateLocator(unittest.TestCase):

    def _make_vehicle_with_plate(self):
        """Synthetic vehicle crop: dark car body + bright plate-shaped rectangle."""
        img = np.full((200, 300, 3), 40, dtype=np.uint8)
        cv2.rectangle(img, (90, 140), (230, 175), (230, 230, 230), -1)
        cv2.rectangle(img, (90, 140), (230, 175), (0, 0, 0), 1)
        return img

    def test_locates_plate_like_region(self):
        locator = OpenCVPlateLocator()
        img = self._make_vehicle_with_plate()
        box = locator.locate(img)
        self.assertIsNotNone(box)
        x1, y1, x2, y2 = box
        aspect = (x2 - x1) / (y2 - y1)
        self.assertTrue(2.0 <= aspect <= 6.0)

    def test_empty_crop_returns_none(self):
        locator = OpenCVPlateLocator()
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertIsNone(locator.locate(empty))

    def test_tiny_crop_returns_none(self):
        locator = OpenCVPlateLocator()
        tiny = np.full((10, 10, 3), 128, dtype=np.uint8)
        self.assertIsNone(locator.locate(tiny))

    def test_blank_image_returns_none(self):
        locator = OpenCVPlateLocator()
        blank = np.full((100, 150, 3), 128, dtype=np.uint8)
        box = locator.locate(blank)
        self.assertIsNone(box)


class TestPlateOCRReaderCleanup(unittest.TestCase):

    def test_clean_strips_non_alphanumeric(self):
        cleaned = PlateOCRReader._clean("AB-12 cd 34!")
        self.assertEqual(cleaned, "AB12CD34")

    def test_clean_uppercases(self):
        cleaned = PlateOCRReader._clean("dl9caf1234")
        self.assertEqual(cleaned, "DL9CAF1234")

    def test_read_handles_empty_crop(self):
        reader = PlateOCRReader()
        result = reader.read(np.zeros((0, 0, 3), dtype=np.uint8))
        self.assertEqual(result.text, "")
        self.assertEqual(result.confidence, 0.0)


class TestPlateRecognitionPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_evidence_plates"
        self.pipeline = PlateRecognitionPipeline(crops_dir=self.test_dir,
                                                  retry_cooldown_frames=10)

    def tearDown(self):
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _make_frame(self):
        return np.full((480, 640, 3), 60, dtype=np.uint8)

    def test_recognize_returns_result_and_saves_evidence(self):
        frame = self._make_frame()
        bbox = [100, 100, 300, 300]
        result = self.pipeline.recognize(frame, bbox, track_id=1, frame_num=10)
        self.assertTrue(os.path.exists(result.image_path))

    def test_invalid_bbox_returns_empty_result(self):
        frame = self._make_frame()
        bbox = [600, 600, 590, 590]
        result = self.pipeline.recognize(frame, bbox, track_id=2, frame_num=10)
        self.assertEqual(result.plate_number, "")
        self.assertEqual(result.image_path, "")

    def test_retry_cooldown_skips_repeat_attempts(self):
        frame = self._make_frame()
        bbox = [100, 100, 300, 300]
        r1 = self.pipeline.recognize(frame, bbox, track_id=3, frame_num=10)
        r2 = self.pipeline.recognize(frame, bbox, track_id=3, frame_num=12)
        self.assertEqual(r2.plate_number, "")
        self.assertEqual(r2.image_path, "")

    def test_confident_result_is_cached(self):
        from .pipeline import PlateRecognitionResult
        self.pipeline._confident_cache[5] = PlateRecognitionResult(
            plate_number="ABC1234", confidence=90.0, image_path="cached.jpg"
        )
        frame = self._make_frame()
        bbox = [100, 100, 300, 300]
        result = self.pipeline.recognize(frame, bbox, track_id=5, frame_num=999)
        self.assertEqual(result.plate_number, "ABC1234")
        self.assertEqual(result.image_path, "cached.jpg")

    def test_forget_track_clears_cache(self):
        from .pipeline import PlateRecognitionResult
        self.pipeline._confident_cache[7] = PlateRecognitionResult(
            plate_number="XYZ9999", confidence=95.0, image_path="old.jpg"
        )
        self.pipeline.forget_track(7)
        frame = self._make_frame()
        result = self.pipeline.recognize(frame, [100, 100, 300, 300], track_id=7, frame_num=1)
        self.assertEqual(result.plate_number, "")


if __name__ == "__main__":
    unittest.main()

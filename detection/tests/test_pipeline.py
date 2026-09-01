"""
Unit tests for the violation rule engine and the Bangladesh plate recognizer.

Run with:
    python -m detection.main --test
or directly:
    python -m unittest detection.tests.test_pipeline -v
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import supervision as sv

from ..config import (
    COOLDOWN_FRAMES, LINE_UPPER_Y, LINE_LOWER_Y, REAL_DISTANCE_METERS,
    VIDEO_FPS, SPEED_LIMIT_KMH, STOP_LINE_Y,
)
from ..models import ViolationType
from ..speed_estimator import SpeedEstimator
from ..redlight_detector import RedLightDetector
from ..rule_engine import ViolationRuleEngine
from ..plate_recognizer import BangladeshPlateRecognizer, OCRCandidate


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




"""
Command-line entry point for the Traffic Violation Detection System.

Usage (run from the project root):
    python -m detection.main                          # use VIDEO_PATH from config.py
    python -m detection.main --video assets/videos/x.mp4
    python -m detection.main --video "rtsp://user:pass@192.168.1.50:554/stream1"
    python -m detection.main --video 0                 # USB webcam index 0
    python -m detection.main --test                    # run unit tests
    python -m detection.main --ocr-image path/to/plate.jpg
"""

from __future__ import annotations

import json
import sys
import unittest
from collections import deque
from pathlib import Path
from typing import Dict

import cv2
import supervision as sv
from ultralytics import YOLO

from .config import (
    PROJECT_ROOT, VIDEO_PATH, MODEL_PATH, VEHICLE_CLASSES, CONFIDENCE, IOU,
    DISPLAY_WIDTH, INFERENCE_IMGSZ, LINE_UPPER_Y, LINE_LOWER_Y, REAL_DISTANCE_METERS,
    VIDEO_FPS, SPEED_LIMIT_KMH, STOP_LINE_Y, USE_KEYBOARD, COOLDOWN_FRAMES,
    EVIDENCE_PRE_FRAMES, EVIDENCE_POST_FRAMES, OCR_RETRY_FRAMES,
    OCR_CONFIDENCE_MIN,
)
from .models import ViolationEvent
from .traffic_light import TrafficLight
from .speed_estimator import SpeedEstimator
from .redlight_detector import RedLightDetector
from .tracking import OcclusionTracker
from .rule_engine import ViolationRuleEngine
from .plate_recognizer import BangladeshPlateRecognizer
from .rendering import draw_hud, draw_direction_indicators, build_labels, resize_for_display
from .video_stream import ThreadedVideoStream

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


#  MAIN


def main():
    import sys
    if "--test" in sys.argv:
        print("Running TVS-9 + Epic 4 unit tests...")
        suite = unittest.TestSuite()
        from .tests.test_pipeline import TestViolationRuleEngine, TestBangladeshPlateRecognizer
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
    from integration import ViolationDBWriter
    db_writer = ViolationDBWriter()
    print(f"Database      : {'CONNECTED' if db_writer.ping() else 'OFFLINE (queue enabled)'}")
    flushed = db_writer.flush_pending()
    if flushed:
        print(f"  [DB] Flushed {flushed} queued event(s)")

    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.55)

    # Live sources (RTSP/HTTP camera streams, webcam indices) are read on a
    # background thread that always keeps only the newest frame — this is
    # what prevents the pipeline from falling further and further behind a
    # live feed when processing can't keep up frame-for-frame. Local video
    # *files* are still read synchronously so every frame gets processed
    # (there's no "falling behind" risk, and we don't want to skip frames
    # when analyzing recorded footage).
    is_live_source = isinstance(video_path, str) and (
        video_path.lower().startswith(("rtsp://", "http://", "https://"))
        or video_path.isdigit()
    )
    if is_live_source:
        source = int(video_path) if video_path.isdigit() else video_path
        cap = ThreadedVideoStream(source).start()
    else:
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
                           conf=CONFIDENCE, iou=IOU, imgsz=INFERENCE_IMGSZ, verbose=False)
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

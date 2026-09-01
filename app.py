"""
app.py
Integrated runner for TVS-9 + TVS-10 with YOLO plate detection.
"""

import os
import sys
import json
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from violation_rule_engine import (
    VIDEO_PATH, MODEL_PATH, VEHICLE_CLASSES, CONFIDENCE, IOU, DISPLAY_WIDTH,
    LINE_UPPER_Y, LINE_LOWER_Y, REAL_DISTANCE_METERS, VIDEO_FPS, SPEED_LIMIT_KMH,
    STOP_LINE_Y, USE_KEYBOARD,
    COOLDOWN_FRAMES, EVIDENCE_PRE_FRAMES, EVIDENCE_POST_FRAMES,
    TrafficLight, OcclusionTracker, SpeedEstimator, RedLightDetector,
    ViolationRuleEngine, SignalState,
    draw_hud, draw_direction_indicators, build_labels, resize_for_display,
)

from plate_module import PlateRecognitionPipeline

OUTPUT_VIDEO_PATH = "output/annotated_vehicles.mp4"


def run_tests():
    import unittest
    print("=" * 60)
    print("Running TVS-9 unit tests...")
    print("=" * 60)
    from violation_rule_engine import TestViolationRuleEngine
    suite = unittest.TestLoader().loadTestsFromTestCase(TestViolationRuleEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    r1 = runner.run(suite)

    print("\n" + "=" * 60)
    print("Running TVS-10 unit tests...")
    print("=" * 60)
    from plate_module.test_plate_module import (
        TestOpenCVPlateLocator,
        TestPlateOCRReaderCleanup,
        TestPlateRecognitionPipeline,
    )
    suite2 = unittest.TestSuite()
    suite2.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOpenCVPlateLocator))
    suite2.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPlateOCRReaderCleanup))
    suite2.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPlateRecognitionPipeline))
    r2 = runner.run(suite2)

    ok = r1.wasSuccessful() and r2.wasSuccessful()
    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
    return 0 if ok else 1


def main():
    no_display = "--no-display" in sys.argv

    if "--test" in sys.argv:
        return run_tests()

    print("=" * 60)
    print("TVS-9 + TVS-10: Violation Rule Engine + Plate Recognition")
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

    model = YOLO(MODEL_PATH)
    tracker = OcclusionTracker(frame_rate=VIDEO_FPS)
    light = TrafficLight()

    speed_est = SpeedEstimator(
        LINE_UPPER_Y, LINE_LOWER_Y,
        REAL_DISTANCE_METERS, VIDEO_FPS, SPEED_LIMIT_KMH
    )
    rl_det = RedLightDetector(STOP_LINE_Y)
    engine = ViolationRuleEngine(speed_est, rl_det, SPEED_LIMIT_KMH)

    plate_pipeline = PlateRecognitionPipeline()
    print("-" * 60)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.55)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Cannot open video:", VIDEO_PATH)
        return 1

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    if not (0 <= LINE_UPPER_Y < LINE_LOWER_Y < orig_h):
        print("ERROR: Speed-line coordinates must be within the video height "
              f"(0-{orig_h - 1}): upper={LINE_UPPER_Y}, lower={LINE_LOWER_Y}")
        cap.release()
        return 1

    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (orig_w, orig_h),
    )
    if not video_writer.isOpened():
        print(f"ERROR: Cannot create output video: {OUTPUT_VIDEO_PATH}")
        cap.release()
        return 1

    print(f"Video         : {orig_w}x{orig_h}")
    print(f"Output video  : {OUTPUT_VIDEO_PATH}")
    if not no_display:
        print("Controls      : Q=quit  P=pause  R=Red  Y=Yellow  G=Green")
    print("=" * 60)

    frame_count = 0
    paused = False

    while True:
        key = cv2.waitKey(1) & 0xFF if not no_display else 0xFF

        if key == ord("q"):
            break
        elif key == ord("p") and not no_display:
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
        results = model(frame, classes=VEHICLE_CLASSES,
                        conf=CONFIDENCE, iou=IOU, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        tracked, lost_ids = tracker.update(detections, frame_count)

        # Clear plate cache for lost tracks
        for lost_id in lost_ids:
            plate_pipeline.forget_track(lost_id)

        # Rule engine
        new_events = engine.update(tracked, light.state, frame_count)

        # Plate recognition for new violations
        for event in new_events:
            if tracked.tracker_id is not None:
                for i, tid in enumerate(tracked.tracker_id):
                    if int(tid) == event.track_id:
                        bbox = tracked.xyxy[i].tolist()
                        plate_result = plate_pipeline.recognize(
                            frame, bbox, event.track_id, frame_count
                        )
                        event.plate_number = plate_result.plate_number
                        event.image_path = plate_result.image_path

                        if plate_result.plate_number:
                            print(f"  [PLATE OCR] Vehicle #{event.track_id}: "
                                  f"'{plate_result.plate_number}' "
                                  f"(conf={plate_result.confidence:.1f})")
                        break

        # Draw scene
        annotated = frame.copy()
        draw_hud(annotated, frame_count, engine, tracked,
                 light.state, orig_w, orig_h)
        draw_direction_indicators(annotated, tracked, engine.rl_detector)

        # ── LIVE plate locator boxes (yellow) ──
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                bbox = tracked.xyxy[i].tolist()
                plate_frame_box = plate_pipeline.preview_plate_box(frame, bbox)
                if plate_frame_box:
                    p_x1, p_y1, p_x2, p_y2 = plate_frame_box
                    cv2.rectangle(annotated, (p_x1, p_y1), (p_x2, p_y2),
                                  (0, 255, 255), 2)
                    cv2.putText(annotated, "PLATE", (p_x1, p_y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1)

        labels = build_labels(tracked, engine, model)
        annotated = box_annotator.annotate(scene=annotated, detections=tracked)
        annotated = label_annotator.annotate(scene=annotated,
                                              detections=tracked,
                                              labels=labels)
        video_writer.write(annotated)

        if no_display:
            if frame_count % 30 == 0:
                print(f"  Frame {frame_count} | Events: {len(engine.events)}")
        else:
            display = resize_for_display(annotated, DISPLAY_WIDTH)
            cv2.imshow("TVS-9 + TVS-10", display)

    cap.release()
    video_writer.release()
    if not no_display:
        cv2.destroyAllWindows()

    # Final Report
    summary = engine.get_summary()
    speed_summary = engine.speed_est.get_summary()
    rl_summary = engine.rl_detector.get_summary()

    print("\n" + "=" * 60)
    print("FINAL REPORT")
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

    print(f"\nRed light hits    : {rl_summary['total_violations']} "
          f"(UP: {rl_summary['up_violations']}, DOWN: {rl_summary['down_violations']})")

    if summary["events"]:
        print("\nEvent log (with plates):")
        for e in summary["events"]:
            speed_str = f"{e['speed_kmh']} km/h" if e['speed_kmh'] else "N/A"
            plate_str = f"plate='{e['plate_number']}'" if e['plate_number'] else "plate=N/A"
            print(f"  [{e['event_id']}]  "
                  f"Frame {e['frame_number']:>5}  "
                  f"Vehicle #{e['track_id']:>3}  "
                  f"{e['violation_type']:<10}  "
                  f"speed={speed_str:<12}  "
                  f"dir={e['direction']:<5}  "
                  f"signal={e['signal_state']}  "
                  f"{plate_str}")

    combined_report = {
        "rule_engine": summary,
        "speed_estimation": speed_summary,
        "red_light_detection": rl_summary,
    }
    out = "violation_events.json"
    with open(out, "w") as f:
        json.dump(combined_report, f, indent=2)
    print(f"\nSaved: {out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

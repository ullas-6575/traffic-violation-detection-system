"""HUD drawing helpers: overlays, direction arrows, and vehicle labels."""

from __future__ import annotations

import cv2

from .config import LINE_UPPER_Y, LINE_LOWER_Y, STOP_LINE_Y, USE_KEYBOARD, EVIDENCE_POST_FRAMES
from .models import Direction, SignalState, ViolationType
from .rule_engine import ViolationRuleEngine
from .redlight_detector import RedLightDetector


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




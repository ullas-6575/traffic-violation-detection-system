"""TVS-9: Violation rule engine — coordinates speed + red-light detectors."""

from __future__ import annotations

from datetime import datetime
from typing import List

import supervision as sv

from .config import COOLDOWN_FRAMES, EVIDENCE_PRE_FRAMES, EVIDENCE_POST_FRAMES
from .models import ViolationEvent, ViolationType, SignalState
from .speed_estimator import SpeedEstimator
from .redlight_detector import RedLightDetector


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




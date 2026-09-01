"""Shared data model: enums and dataclasses used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict


class Direction(Enum):
    UNKNOWN = "UNKNOWN"
    UP      = "UP"       # Y decreasing — enters from bottom
    DOWN    = "DOWN"     # Y increasing — enters from top


class SignalState(Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    RED    = "RED"


class ViolationType(str, Enum):
    OVERSPEED  = "OVERSPEED"
    RED_LIGHT  = "RED_LIGHT"


@dataclass
class SpeedMeasurement:
    """Single speed measurement result (TVS-7)."""
    track_id: int
    direction: str
    speed_kmh: float
    violation: bool
    start_frame: int
    end_frame: int
    frames_between: float
    time_seconds: float


@dataclass
class ViolationEvent:
    """
    Canonical violation record emitted by the rule engine.
    Consumed downstream by: plate crop → OCR → MySQL writer → Laravel dashboard.
    """
    event_id:             str              # unique: "{track_id}_{type}_{frame}_{counter}"
    track_id:             int
    violation_type:       ViolationType
    frame_number:         int
    timestamp:            str              # ISO 8601
    direction:            str              # "up" / "down" / "UP" / "DOWN" / "unknown"
    signal_state:         str              # "RED" / "GREEN" / "YELLOW" / "N/A"
    speed_kmh:            Optional[float]  # None for red-light-only events
    speed_limit_kmh:      Optional[float]
    bbox:                 list             # [x1, y1, x2, y2] at violation frame
    evidence_start_frame: int              # clip window start
    evidence_end_frame:   int              # clip window end
    plate_number:         str = ""         # filled by TVS-10/11
    image_path:           str = ""         # filled by TVS-10
    plate_crop_path:      str = ""         # cropped/enhanced plate evidence
    ocr_raw_text:         str = ""         # unmodified winning OCR output
    ocr_confidence:       float = 0.0       # normalized 0..1
    ocr_engine:           str = ""         # easyocr / tesseract / none
    vehicle_color:        str = "UNKNOWN"  # estimated body color
    color_confidence:     float = 0.0       # normalized 0..1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["violation_type"] = self.violation_type.value
        return d




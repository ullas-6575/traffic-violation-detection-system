"""Manual (keyboard) or auto-cycling traffic-light state."""

from __future__ import annotations

from .config import LIGHT_CYCLE_FRAMES, GREEN_FRAMES, YELLOW_FRAMES
from .models import SignalState


class TrafficLight:
    def __init__(self):
        self.state = SignalState.GREEN

    def set_state(self, key: int):
        if key == ord('r'):
            self.state = SignalState.RED
            print("  Signal → RED")
        elif key == ord('y'):
            self.state = SignalState.YELLOW
            print("  Signal → YELLOW")
        elif key == ord('g'):
            self.state = SignalState.GREEN
            print("  Signal → GREEN")

    def auto_update(self, frame_num: int):
        pos = frame_num % LIGHT_CYCLE_FRAMES
        if pos < GREEN_FRAMES:
            self.state = SignalState.GREEN
        elif pos < GREEN_FRAMES + YELLOW_FRAMES:
            self.state = SignalState.YELLOW
        else:
            self.state = SignalState.RED

    def get_bgr(self) -> tuple:
        return {
            SignalState.GREEN:  (0, 255, 0),
            SignalState.YELLOW: (0, 255, 255),
            SignalState.RED:    (0, 0, 255),
        }[self.state]




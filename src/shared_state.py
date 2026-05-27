from dataclasses import dataclass, asdict
from threading import Lock


@dataclass
class AppState:
    cars_left: int = 0
    cars_right: int = 0
    active_model: str = "none"
    detection_enabled: bool = False
    camera_error: str | None = None


class SharedState:
    """Small thread-safe container for values shared with the web UI."""

    def __init__(self):
        self._state = AppState()
        self._lock = Lock()

    def set_camera_error(self, error):
        with self._lock:
            self._state.camera_error = error

    def to_dict(self):
        with self._lock:
            return asdict(self._state)

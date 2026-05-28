from src.detection.base import BaseDetector


class NoneDetector(BaseDetector):
    """Detector used when object detection is disabled."""

    name = "none"
    display_name = "none"
    backend = "none"
    model_path = None
    enabled = False

    def detect(self, frame):
        return []

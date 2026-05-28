from src.detection.base import BaseDetector


class NoneDetector(BaseDetector):
    """Detector used when object detection is disabled."""

    name = "none"
    enabled = False

    def detect(self, frame):
        return []

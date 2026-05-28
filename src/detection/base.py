class BaseDetector:
    """Common detector interface used by the camera pipeline."""

    name = "none"
    enabled = False

    def detect(self, frame):
        """Return vehicle detections as dictionaries with class/confidence/bbox."""
        return []

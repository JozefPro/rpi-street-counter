class LineCounter:
    """Placeholder counter state for future line-crossing tracking."""

    def __init__(self):
        self.cars_left = 0
        self.cars_right = 0

    def update(self, detections):
        # TODO: Count tracked cars crossing configured road lines.
        return {
            "cars_left": self.cars_left,
            "cars_right": self.cars_right,
        }

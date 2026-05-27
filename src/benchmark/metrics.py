class BenchmarkMetrics:
    """Placeholder for future model/runtime benchmark data."""

    def __init__(self):
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)

    def summary(self):
        return {"samples": len(self.samples)}

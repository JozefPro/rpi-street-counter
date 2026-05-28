from src.detection.base import BaseDetector


class YoloUltralyticsDetector(BaseDetector):
    """Vehicle detector backed by an Ultralytics YOLO model."""

    enabled = True

    def __init__(self, name, weights, confidence_threshold, class_names, input_size):
        self.name = name
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.class_names = set(class_names)
        self.input_size = input_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        from ultralytics import YOLO

        print(f"Loading detector model {self.name} from {self.weights}", flush=True)
        self._model = YOLO(self.weights)

    def detect(self, frame):
        self._load_model()
        results = self._model.predict(
            frame,
            imgsz=self.input_size,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                class_name = names.get(class_id, str(class_id))
                if class_name not in self.class_names:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": round(float(box.conf[0]), 3),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    }
                )

        return detections

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

from src.detection.base import BaseDetector


COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class OpenCVDnnYoloDetector(BaseDetector):
    """YOLOv5 ONNX detector using OpenCV DNN, avoiding a PyTorch dependency."""

    enabled = True

    def __init__(
        self,
        name,
        display_name,
        backend,
        weights,
        model_url,
        confidence_threshold,
        class_names,
        input_size,
        input_width=None,
        input_height=None,
        fixed_input_size=False,
    ):
        self.name = name
        self.display_name = display_name
        self.backend = backend
        self.weights = Path(weights)
        self.model_path = str(self.weights)
        self.model_url = model_url
        self.confidence_threshold = confidence_threshold
        self.class_names = set(class_names)
        self.input_size = int(input_size)
        self.input_width = int(input_width or input_size)
        self.input_height = int(input_height or input_size)
        self.fixed_input_size = bool(fixed_input_size)
        self.effective_input_width = self.input_size if self.fixed_input_size else self.input_width
        self.effective_input_height = self.input_size if self.fixed_input_size else self.input_height
        self._net = None
        self._shape_fallback_logged = False

    def _load_model(self):
        if self._net is not None:
            return

        if not self.weights.exists():
            if not self.model_url:
                raise FileNotFoundError(f"Detector weights not found: {self.weights}")
            self.weights.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {self.display_name} model to {self.weights}...", flush=True)
            urlretrieve(self.model_url, self.weights)
        else:
            print(f"Using local model: {self.weights}", flush=True)

        print(f"Loading {self.display_name} with {self.backend} from {self.weights}", flush=True)
        if self.fixed_input_size:
            print(
                f"{self.display_name} ONNX uses fixed model input "
                f"{self.input_size}x{self.input_size}",
                flush=True,
            )
        self._net = cv2.dnn.readNetFromONNX(str(self.weights))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, frame):
        self._load_model()

        frame_height, frame_width = frame.shape[:2]
        blob = self._make_blob(frame, self.effective_input_width, self.effective_input_height)
        self._net.setInput(blob)
        try:
            output = self._net.forward()
            model_input_width = self.effective_input_width
            model_input_height = self.effective_input_height
        except cv2.error:
            if self.effective_input_width == self.input_size and self.effective_input_height == self.input_size:
                raise

            if not self._shape_fallback_logged:
                print(
                    "OpenCV DNN model rejected configured inference shape "
                    f"{self.input_width}x{self.input_height}; "
                    f"falling back to {self.input_size}x{self.input_size}",
                    flush=True,
                )
                self._shape_fallback_logged = True

            self.effective_input_width = self.input_size
            self.effective_input_height = self.input_size
            blob = self._make_blob(frame, self.input_size, self.input_size)
            self._net.setInput(blob)
            output = self._net.forward()
            model_input_width = self.input_size
            model_input_height = self.input_size

        raw_predictions = np.squeeze(output)
        if len(raw_predictions.shape) != 2:
            raw_predictions = raw_predictions.reshape(-1, raw_predictions.shape[-1])

        if raw_predictions.shape[0] in (84, 85):
            predictions = raw_predictions.T
        else:
            predictions = raw_predictions
        boxes = []
        confidences = []
        class_ids = []

        x_scale = frame_width / model_input_width
        y_scale = frame_height / model_input_height

        for prediction in predictions:
            if len(prediction) >= 85:
                objectness = float(prediction[4])
                scores = prediction[5:]
            else:
                objectness = 1.0
                scores = prediction[4:]

            class_id = int(np.argmax(scores))
            if class_id >= len(COCO_CLASS_NAMES):
                continue
            confidence = objectness * float(scores[class_id])
            class_name = COCO_CLASS_NAMES[class_id]

            if confidence < self.confidence_threshold or class_name not in self.class_names:
                continue

            center_x, center_y, width, height = prediction[:4]
            x1 = int((center_x - width / 2) * x_scale)
            y1 = int((center_y - height / 2) * y_scale)
            box_width = int(width * x_scale)
            box_height = int(height * y_scale)

            boxes.append([x1, y1, box_width, box_height])
            confidences.append(confidence)
            class_ids.append(class_id)

        keep_indexes = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            0.45,
        )

        detections = []
        for index in np.array(keep_indexes).flatten():
            x, y, width, height = boxes[index]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_width - 1, x + width)
            y2 = min(frame_height - 1, y + height)
            class_name = COCO_CLASS_NAMES[class_ids[index]]

            detections.append(
                {
                    "class_name": class_name,
                    "confidence": round(confidences[index], 3),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections

    def _make_blob(self, frame, input_width, input_height):
        return cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=(input_width, input_height),
            swapRB=True,
            crop=False,
        )

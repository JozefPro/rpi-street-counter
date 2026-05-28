from src.detection.none_detector import NoneDetector


def create_detector(config):
    detection_config = config.get("detection", {})

    if not detection_config.get("enabled", False):
        return NoneDetector()

    model_name = detection_config.get("model", "none")
    if model_name == "none":
        return NoneDetector()

    models = config.get("models", {})
    model_config = models.get(model_name)
    if not model_config:
        raise ValueError(f"Unknown detection model: {model_name}")

    model_type = model_config.get("type")

    # MODEL SELECTION HAPPENS HERE.
    # Change detection.model in config.yaml to choose another model.
    if model_type == "opencv_dnn_yolo":
        from src.detection.opencv_dnn import OpenCVDnnYoloDetector

        return OpenCVDnnYoloDetector(
            name=model_name,
            display_name=model_config.get("display_name", model_name),
            backend=model_config.get("backend", "OpenCV DNN / ONNX"),
            weights=model_config["weights"],
            model_url=model_config.get("url"),
            confidence_threshold=detection_config.get("confidence_threshold", 0.35),
            class_names=detection_config.get("classes", []),
            input_size=detection_config.get("input_size", 320),
            input_width=detection_config.get("inference_width"),
            input_height=detection_config.get("inference_height"),
            fixed_input_size=model_config.get("fixed_input_size", False),
        )

    if model_type == "ultralytics":
        from src.detection.yolo_ultralytics import YoloUltralyticsDetector

        return YoloUltralyticsDetector(
            name=model_name,
            weights=model_config["weights"],
            confidence_threshold=detection_config.get("confidence_threshold", 0.35),
            class_names=detection_config.get("classes", []),
            input_size=detection_config.get("input_size", 320),
        )

    raise ValueError(f"Unsupported detection model type: {model_type}")

from copy import deepcopy
from pathlib import Path
import time

import yaml
from flask import Flask, Response, jsonify, render_template

from src.camera import CameraReader
from src.detection import create_detector
from src.shared_state import SharedState
from src.system_stats import get_system_stats


DEFAULT_CONFIG = {
    "camera": {"index": 0, "width": 1280, "height": 720, "fps": 30},
    "server": {"host": "0.0.0.0", "port": 5000, "debug": False},
    "stream": {"jpeg_quality": 70, "max_stream_fps": 15},
    "detection": {
        "enabled": True,
        "model": "yolo_nano",
        "confidence_threshold": 0.35,
        "classes": ["car", "truck", "bus", "motorcycle"],
        "run_every_n_frames": 3,
        "input_size": 640,
    },
    "models": {
        "yolo_nano": {
            "type": "opencv_dnn_yolo",
            "weights": "models/yolov5n.onnx",
            "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx",
        },
    },
    "project": {"name": "RPI5 Street Counter"},
}


def load_config(path="config.yaml"):
    config = deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        for section, values in loaded.items():
            if isinstance(values, dict) and isinstance(config.get(section), dict):
                config[section] = {**config[section], **values}
            else:
                config[section] = values

    return config


config = load_config()
shared_state = SharedState()
detector = create_detector(config)
camera = CameraReader(
    index=config["camera"]["index"],
    width=config["camera"]["width"],
    height=config["camera"]["height"],
    target_fps=config["camera"]["fps"],
    jpeg_quality=config["stream"]["jpeg_quality"],
    max_stream_fps=config["stream"]["max_stream_fps"],
    detector=detector,
    detection_enabled=config["detection"]["enabled"],
    detection_run_every_n_frames=config["detection"]["run_every_n_frames"],
)

app = Flask(__name__)


@app.before_request
def ensure_camera_started():
    camera.start()


@app.route("/")
def index():
    return render_template(
        "index.html",
        project_name=config["project"]["name"],
    )


def mjpeg_frames():
    frame_interval = 1.0 / max(1, int(config["stream"]["max_stream_fps"]))

    while True:
        frame = camera.get_jpeg_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(frame_interval)


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    shared_state.set_camera_error(camera.error)
    state = shared_state.to_dict()
    stats = get_system_stats()

    return jsonify(
        {
            **state,
            **stats,
            "requested_width": camera.width,
            "requested_height": camera.height,
            "requested_fps": camera.target_fps,
            "actual_width": camera.actual_width,
            "actual_height": camera.actual_height,
            "actual_camera_fps": camera.actual_camera_fps,
            "measured_camera_fps": camera.camera_fps,
            "measured_stream_fps": camera.stream_fps,
            "stream_max_fps": camera.max_stream_fps,
            "jpeg_quality": camera.jpeg_quality,
            "fps": camera.stream_fps,
            "camera_running": camera.running,
            "detection_enabled": camera.detection_enabled,
            "active_model": camera.active_model,
            "inference_ms": camera.inference_ms,
            "detection_fps": camera.detection_fps,
            "vehicle_detections_count": camera.vehicle_detections_count,
            "detection_error": camera.detection_error,
        }
    )


if __name__ == "__main__":
    camera.start()
    app.run(
        host=config["server"]["host"],
        port=config["server"]["port"],
        debug=config["server"]["debug"],
        threaded=True,
        use_reloader=False,
    )

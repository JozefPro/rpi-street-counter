from copy import deepcopy
from pathlib import Path
import time

import yaml
from flask import Flask, Response, jsonify, render_template

from src.camera import CameraReader
from src.detection import create_detector
from src.shared_state import SharedState
from src.system_stats import get_system_stats
from src.tracking.line_counter import LineCounter


DEFAULT_CONFIG = {
    "camera": {"index": 0, "width": 1280, "height": 720, "fps": 30},
    "server": {"host": "0.0.0.0", "port": 5000, "debug": False},
    "stream": {"jpeg_quality": 70, "max_stream_fps": 30, "delay_frames": 4},
    "detection": {
        "enabled": True,
        "model": "yolo_nano",
        "confidence_threshold": 0.35,
        "classes": ["car", "truck", "bus", "motorcycle"],
        "run_every_n_frames": 3,
        "inference_width": 640,
        "inference_height": 360,
        "input_size": 640,
    },
    "models": {
        "yolo_nano": {
            "type": "opencv_dnn_yolo",
            "weights": "models/yolov5n.onnx",
            "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx",
        },
    },
    "counting": {
        "enabled": True,
        "draw_lines": True,
        "label_offset_px": [-28, -14],
        "counted_classes": ["car", "truck", "bus", "motorcycle", "motorbike"],
        "line_a": {
            "name": "A",
            "p1_norm": [0.14, 0.66],
            "p2_norm": [0.69, 0.28],
            "color": [0, 0, 255],
        },
        "line_b": {
            "name": "B",
            "p1_norm": [0.42, 0.98],
            "p2_norm": [0.94, 0.47],
            "color": [0, 0, 255],
        },
        "sequence_a_then_b": "left",
        "sequence_b_then_a": "right",
        "max_track_age_seconds": 6.0,
        "counter_window_seconds": 300,
    },
    "tracking": {
        "max_distance_px": 190,
        "max_age_seconds": 6.0,
    },
    "debug": {
        "draw_track_centers": True,
        "draw_track_ids": True,
        "draw_movement_segments": False,
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
line_counter = LineCounter(
    config.get("counting", {}),
    config["detection"].get("classes", []),
    tracking_config=config.get("tracking", {}),
    debug_config=config.get("debug", {}),
)
camera = CameraReader(
    index=config["camera"]["index"],
    width=config["camera"]["width"],
    height=config["camera"]["height"],
    target_fps=config["camera"]["fps"],
    jpeg_quality=config["stream"]["jpeg_quality"],
    max_stream_fps=config["stream"]["max_stream_fps"],
    delay_frames=config["stream"].get("delay_frames", 0),
    detector=detector,
    detection_enabled=config["detection"]["enabled"],
    detection_run_every_n_frames=config["detection"]["run_every_n_frames"],
    detection_inference_width=config["detection"].get("inference_width"),
    detection_inference_height=config["detection"].get("inference_height"),
    line_counter=line_counter,
    counter_window_seconds=config["counting"].get("counter_window_seconds", 300),
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
    count_window_status = camera.get_count_window_status()

    return jsonify(
        {
            **state,
            **stats,
            **count_window_status,
            "requested_width": camera.width,
            "requested_height": camera.height,
            "requested_fps": camera.target_fps,
            "actual_width": camera.actual_width,
            "actual_height": camera.actual_height,
            "actual_camera_fps": camera.actual_camera_fps,
            "measured_camera_fps": camera.camera_fps,
            "measured_stream_fps": camera.stream_fps,
            "stream_max_fps": camera.max_stream_fps,
            "stream_delay_frames": camera.delay_frames,
            "latest_frame_id": camera.latest_frame_id,
            "streamed_frame_id": camera.streamed_frame_id,
            "detection_frame_id": camera.detection_frame_id,
            "frame_buffer_size": camera.frame_buffer_size,
            "jpeg_quality": camera.jpeg_quality,
            "fps": camera.stream_fps,
            "camera_running": camera.running,
            "detection_enabled": camera.detection_enabled,
            "active_model": camera.active_model,
            "inference_ms": camera.inference_ms,
            "inference_width": camera.inference_width,
            "inference_height": camera.inference_height,
            "inference_frame_width": camera.inference_frame_width,
            "inference_frame_height": camera.inference_frame_height,
            "detection_run_every_n_frames": camera.detection_run_every_n_frames,
            "detection_fps": camera.detection_fps,
            "vehicle_detections_count": camera.vehicle_detections_count,
            "boxes_drawn_count": camera.boxes_drawn_count,
            "stream_uses_annotated_frame": camera.stream_uses_annotated_frame,
            "detection_error": camera.detection_error,
            "cars_left": camera.cars_left,
            "cars_right": camera.cars_right,
            "total_counted": camera.total_counted,
            "counting_enabled": camera.counting_enabled,
            "line_a": camera.line_a,
            "line_b": camera.line_b,
            "active_tracks": camera.active_tracks,
            "latest_crossing_event": camera.latest_crossing_event,
            "line_a_crossings_seen": camera.line_a_crossings_seen,
            "line_b_crossings_seen": camera.line_b_crossings_seen,
            "tracks_waiting_for_second_line": camera.tracks_waiting_for_second_line,
            "track_id_switches": camera.track_id_switches,
            "counted_classes": camera.counted_classes,
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

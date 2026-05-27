from copy import deepcopy
from pathlib import Path
import time

import yaml
from flask import Flask, Response, jsonify, render_template

from src.camera import CameraReader
from src.shared_state import SharedState
from src.system_stats import get_system_stats


DEFAULT_CONFIG = {
    "camera": {"index": 0, "width": 1280, "height": 720, "fps": 30},
    "server": {"host": "0.0.0.0", "port": 5000, "debug": False},
    "stream": {"jpeg_quality": 80},
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
camera = CameraReader(
    index=config["camera"]["index"],
    width=config["camera"]["width"],
    height=config["camera"]["height"],
    target_fps=config["camera"]["fps"],
    jpeg_quality=config["stream"]["jpeg_quality"],
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
    while True:
        frame = camera.get_jpeg_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.001)


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
            "fps": camera.fps,
            "camera_running": camera.running,
        }
    )


if __name__ == "__main__":
    camera.start()
    app.run(
        host=config["server"]["host"],
        port=config["server"]["port"],
        debug=config["server"]["debug"],
        threaded=True,
    )

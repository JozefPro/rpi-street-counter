# RPI5 Street Counter

RPI5 Street Counter is a local LAN web dashboard for a Raspberry Pi 5 with a USB webcam. The final project will detect cars and count vehicles crossing two road lines, but this first milestone only implements the live camera stream foundation.

Current milestone:

- Flask web app served on the Raspberry Pi.
- OpenCV USB webcam reader running in a background thread.
- MJPEG live stream at `/video_feed`.
- JSON status endpoint at `/api/status`.
- Dark dashboard with FPS, CPU, RAM, temperature, and placeholder counters.
- No object detection yet.

## Project Structure

```text
rpi-street-counter/
  app.py
  config.yaml
  requirements.txt
  README.md
  src/
    camera.py
    shared_state.py
    system_stats.py
    detection/base.py
    tracking/line_counter.py
    benchmark/metrics.py
  templates/index.html
  static/style.css
  static/app.js
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Start the Flask app:

```bash
python app.py
```

Open the dashboard on the Raspberry Pi:

```text
http://localhost:5000
```

To access it from another device on the same LAN, find the Raspberry Pi IP address:

```bash
hostname -I
```

Then open:

```text
http://<raspberry-pi-ip>:5000
```

## Configuration

Edit `config.yaml` to change the camera index, resolution, target FPS, JPEG quality, host, port, or debug mode.

## Notes

This milestone intentionally does not include YOLO, MobileNet, tracking, line crossing, benchmarking, nginx, Docker, Tailscale, or systemd. Those parts will be added later on top of this structure.

# RPI5 Street Counter

RPI5 Street Counter is a local LAN web dashboard for a Raspberry Pi 5 with a USB webcam. The final project will count vehicles crossing two road lines; the current milestone streams the camera and can draw vehicle detections.

Current milestone:

- Flask web app served on the Raspberry Pi.
- OpenCV USB webcam reader running in a background thread.
- MJPEG live stream at `/video_feed`.
- JSON status endpoint at `/api/status`.
- Dark dashboard with video, car counters, detection status, and Raspberry Pi stats.
- Optional YOLO nano vehicle detection for drawing bounding boxes.
- Diagonal A/B line counting for vehicles that cross both configured lines.

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
    detection/factory.py
    detection/base.py
    detection/none_detector.py
    detection/yolo_ultralytics.py
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

## Camera resolution

The default requested camera resolution is `1280x720` at 30 FPS with MJPEG output capped to 15 FPS at JPEG quality 70. This keeps the Flask live stream responsive on the Raspberry Pi while leaving CPU headroom for later detection work.

The actual resolution depends on what the USB webcam and Linux driver support, so the status API and dashboard show the applied camera resolution after OpenCV opens the device. `1920x1080` can be enabled later in `config.yaml` for benchmarking, but it is heavier and may stutter when streamed as MJPEG.

## Object detection model selection

Detection is configured in `config.yaml`.

To disable detection:

```yaml
detection:
  enabled: false
  model: "none"
```

To enable YOLO nano:

```yaml
detection:
  enabled: true
  model: "yolo_nano"
```

Model selection happens in `src/detection/factory.py`. The default lightweight implementation is in `src/detection/opencv_dnn.py` and runs a YOLOv8 nano ONNX model through OpenCV DNN, avoiding a PyTorch runtime on the Raspberry Pi.

The default YOLO model is `yolov5n.onnx`, run every 3 camera frames at input size 640. The model file is downloaded on first use to `models/yolov5n.onnx`. It detects road vehicle classes configured under `detection.classes`.

If the Raspberry Pi becomes slow, reduce:

- camera resolution in `config.yaml`
- `stream.jpeg_quality`
- `detection.input_size`
- detection frequency by increasing `detection.run_every_n_frames`

## Stream delay / detection sync

Object detection takes time, so boxes can look slightly out of sync with the newest camera frame. The stream keeps a short frame buffer and can display a frame a few camera frames behind the latest frame so detection overlays line up better.

Configure the delay in `config.yaml`:

```yaml
stream:
  delay_frames: 4
```

`delay_frames: 0` streams the newest buffered frame. Start with `4`; if boxes appear ahead or behind the video, try `0`, `2`, `4`, `6`, or `8`. Higher delay can improve sync but adds visible latency.

## Diagonal counting lines

Vehicle counting is configured in `config.yaml` under `counting`.

Each line is a two-point diagonal segment using normalized coordinates from `0.0` to `1.0`, so the same config works when the camera resolution changes:

```yaml
counting:
  enabled: true
  draw_lines: true
  label_offset_px: [-28, -14]
  counted_classes:
    - car
    - truck
    - bus
    - motorcycle
    - motorbike
  line_a:
    p1_norm: [0.14, 0.66]
    p2_norm: [0.69, 0.28]
  line_b:
    p1_norm: [0.42, 0.98]
    p2_norm: [0.94, 0.47]
```

`p1_norm` and `p2_norm` define the endpoints of each diagonal line. The app converts them to pixel coordinates for the current frame size and draws labels `A` and `B` on the stream.

`counted_classes` controls which detected vehicle classes can update the counters. The default counts `car`, `truck`, `bus`, `motorcycle`, and `motorbike`; `motorbike` is normalized to `motorcycle` for COCO-style model labels. Bicycle is intentionally not counted by default.

Counting is sequence based. If one tracked vehicle crosses line A and then line B, the direction comes from:

```yaml
sequence_a_then_b: "left"
```

If it crosses line B and then line A, the direction comes from:

```yaml
sequence_b_then_a: "right"
```

If your count direction is reversed, swap those two values. The first tracker is intentionally simple: it matches detections by nearest bounding-box center and expires tracks after `max_track_age_seconds`.

## 5-minute counter windows

The dashboard's large vehicle counters update live, but they represent only the current 5-minute window. When the window ends, the window counters reset to `0` and start counting the next window.

Configure the reset interval in `config.yaml`:

```yaml
counting:
  counter_window_seconds: 300
```

The lifetime counters continue increasing while the app is running and are not reset by the 5-minute window. The dashboard shows the current window start time, last reset time, and countdown to the next reset.

## Counting line tuning

If vehicles are missed, tune the line and tracker values in `config.yaml`.

Move the lines toward the center of the road when vehicles are detected too late or disappear before crossing the second line. Increase `counting.max_track_age_seconds` if tracks disappear too quickly. Increase `tracking.max_distance_px` if track IDs change too often between detector updates.

Debug drawing can help show what the tracker is doing:

```yaml
debug:
  draw_track_centers: true
  draw_track_ids: true
  draw_movement_segments: false
```

`draw_track_centers` and `draw_track_ids` are useful during tuning. `draw_movement_segments` is noisier, but it can show whether a track movement segment actually intersects a counting line.

## Deploy to Raspberry Pi

The Raspberry Pi runs the actual app because the USB webcam is connected there. The Mac is used for development, GitHub, and deployment over SSH.

Make the deploy script executable once:

```bash
chmod +x scripts/deploy_to_pi.sh
```

Deploy the latest committed code to the Raspberry Pi:

```bash
./scripts/deploy_to_pi.sh
```

Deploy and start the Flask app immediately:

```bash
./scripts/deploy_to_pi.sh --run
```

When the app is running, open it from a browser on the same LAN:

```text
http://<rpi-ip>:5000
```

If you started the app with `--run`, stop it in the terminal with `Ctrl+C`.

The deploy script is intentionally conservative: it refuses to deploy if your local repo has uncommitted changes, pushes committed code to GitHub, clones the repo only if the remote directory does not exist, and otherwise updates the existing git checkout without deleting folders.

## Notes

This milestone intentionally does not include final production tracking, benchmarking, Docker, Tailscale changes, or systemd changes. Those parts can be added later on top of this structure.

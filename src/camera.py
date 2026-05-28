import threading
import time

import cv2
import numpy as np


class CameraReader:
    """Owns the webcam and continuously stores the latest frame."""

    def __init__(
        self,
        index=0,
        width=1280,
        height=720,
        target_fps=30,
        jpeg_quality=70,
        max_stream_fps=15,
        detector=None,
        detection_enabled=False,
        detection_run_every_n_frames=3,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.max_stream_fps = max(1, int(max_stream_fps))
        self.detector = detector
        self.detection_enabled = bool(detection_enabled and detector and detector.enabled)
        self.detection_run_every_n_frames = max(1, int(detection_run_every_n_frames))

        self._capture = None
        self._frame = None
        self._jpeg_frame = None
        self._detections = []
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()

        self.fps = 0.0
        self.camera_fps = 0.0
        self.stream_fps = 0.0
        self.actual_width = None
        self.actual_height = None
        self.actual_camera_fps = None
        self.active_model = detector.name if detector else "none"
        self.inference_ms = None
        self.detection_fps = 0.0
        self.vehicle_detections_count = 0
        self.detection_error = None
        self.error = None
        self.running = False

    def start(self):
        with self._start_lock:
            if self._thread and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._capture:
            self._capture.release()
        self.running = False

    def _read_loop(self):
        self._capture = cv2.VideoCapture(self.index)
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)

        if not self._capture.isOpened():
            self.error = f"Could not open camera index {self.index}"
            self._capture.release()
            self.running = False
            return

        self.actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.actual_camera_fps = round(self._capture.get(cv2.CAP_PROP_FPS) or 0, 1)
        print(
            "Camera requested: "
            f"{self.width}x{self.height} at {self.target_fps} FPS; "
            "actual: "
            f"{self.actual_width}x{self.actual_height} at {self.actual_camera_fps} FPS",
            flush=True,
        )

        self.error = None
        self.running = True
        frames = 0
        encoded_frames = 0
        detected_frames = 0
        frame_number = 0
        last_fps_time = time.monotonic()
        next_encode_time = 0.0
        stream_interval = 1.0 / self.max_stream_fps

        while not self._stop_event.is_set():
            ok, frame = self._capture.read()
            if not ok:
                self.error = "Camera frame read failed"
                time.sleep(0.1)
                continue

            self.error = None
            with self._lock:
                self._frame = frame

            frame_number += 1
            if self.detection_enabled and frame_number % self.detection_run_every_n_frames == 0:
                self._run_detection(frame)
                detected_frames += 1

            now = time.monotonic()
            if next_encode_time == 0.0:
                next_encode_time = now

            if now >= next_encode_time:
                jpeg_frame = self._encode_frame(self._draw_detections(frame))
                if jpeg_frame is not None:
                    with self._lock:
                        self._jpeg_frame = jpeg_frame
                    encoded_frames += 1
                next_encode_time += stream_interval
                if next_encode_time < now:
                    next_encode_time = now + stream_interval

            frames += 1
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                self.camera_fps = round(frames / elapsed, 1)
                self.stream_fps = round(encoded_frames / elapsed, 1)
                self.detection_fps = round(detected_frames / elapsed, 1)
                self.fps = self.stream_fps
                frames = 0
                encoded_frames = 0
                detected_frames = 0
                last_fps_time = now

        self._capture.release()
        self.running = False

    def get_latest_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def get_jpeg_frame(self):
        with self._lock:
            if self._jpeg_frame is not None:
                return self._jpeg_frame

        return self._encode_frame(self._placeholder_frame())

    def _run_detection(self, frame):
        started_at = time.monotonic()
        try:
            detections = self.detector.detect(frame)
        except Exception as exc:
            self.detection_error = str(exc)
            print(f"Detection error: {exc}", flush=True)
            return

        elapsed_ms = (time.monotonic() - started_at) * 1000
        with self._lock:
            self._detections = detections

        self.inference_ms = round(elapsed_ms, 1)
        self.vehicle_detections_count = len(detections)
        self.detection_error = None

    def _draw_detections(self, frame):
        with self._lock:
            detections = list(self._detections)

        if not detections:
            return frame

        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (57, 208, 163), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (57, 208, 163),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def _encode_frame(self, frame):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            self.error = "Could not encode camera frame as JPEG"
            return None
        return buffer.tobytes()

    def _placeholder_frame(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        message = self.error or "Waiting for camera..."
        cv2.putText(
            frame,
            message,
            (60, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        return frame

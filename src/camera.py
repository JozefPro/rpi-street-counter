import threading
import time

import cv2
import numpy as np


class CameraReader:
    """Owns the webcam and continuously stores the latest frame."""

    def __init__(self, index=0, width=1280, height=720, target_fps=30, jpeg_quality=80):
        self.index = index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality

        self._capture = None
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()

        self.fps = 0.0
        self.error = None
        self.running = False

    def start(self):
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
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)

        if not self._capture.isOpened():
            self.error = f"Could not open camera index {self.index}"
            self._capture.release()
            self.running = False
            return

        self.error = None
        self.running = True
        frames = 0
        last_fps_time = time.monotonic()

        while not self._stop_event.is_set():
            ok, frame = self._capture.read()
            if not ok:
                self.error = "Camera frame read failed"
                time.sleep(0.1)
                continue

            self.error = None
            with self._lock:
                self._frame = frame

            frames += 1
            now = time.monotonic()
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                self.fps = round(frames / elapsed, 1)
                frames = 0
                last_fps_time = now

        self._capture.release()
        self.running = False

    def get_latest_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def get_jpeg_frame(self):
        frame = self.get_latest_frame()
        if frame is None:
            frame = self._placeholder_frame()

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

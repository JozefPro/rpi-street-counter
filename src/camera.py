from collections import deque
from datetime import datetime
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
        delay_frames=0,
        detector=None,
        detection_enabled=False,
        detection_run_every_n_frames=3,
        detection_inference_width=None,
        detection_inference_height=None,
        line_counter=None,
        counter_window_seconds=300,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.max_stream_fps = max(1, int(max_stream_fps))
        self.delay_frames = max(0, int(delay_frames))
        self.frame_buffer_size = max(self.delay_frames + 10, 30)
        self.detector = detector
        self.detection_enabled = bool(detection_enabled and detector and detector.enabled)
        self.detection_run_every_n_frames = max(1, int(detection_run_every_n_frames))
        self.inference_width = int(detection_inference_width or width)
        self.inference_height = int(detection_inference_height or height)
        self.line_counter = line_counter
        self.counting_enabled = bool(line_counter and line_counter.enabled)
        self.counter_window_seconds = max(1, int(counter_window_seconds))

        self._capture = None
        self._frame = None
        self._jpeg_frame = None
        self._frame_buffer = deque(maxlen=self.frame_buffer_size)
        self._lock = threading.Lock()
        self._detection_lock = threading.Lock()
        self._counter_window_lock = threading.Lock()
        self._detection_busy = False
        self._completed_detections = 0
        self._start_lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._last_debug_log_time = 0.0

        self.fps = 0.0
        self.camera_fps = 0.0
        self.stream_fps = 0.0
        self.actual_width = None
        self.actual_height = None
        self.actual_camera_fps = None
        self.latest_frame_id = 0
        self.streamed_frame_id = None
        self.detection_frame_id = None
        self.model_key = detector.name if detector else "none"
        self.active_model = getattr(detector, "display_name", self.model_key)
        self.model_backend = getattr(detector, "backend", "none")
        self.model_path = getattr(detector, "model_path", None)
        self.inference_ms = None
        self.inference_frame_width = None
        self.inference_frame_height = None
        self.model_input_width = None
        self.model_input_height = None
        self.detection_fps = 0.0
        self.vehicle_detections_count = 0
        self.boxes_drawn_count = 0
        self.stream_uses_annotated_frame = False
        self.detection_error = None
        self.cars_left = 0
        self.cars_right = 0
        self.total_counted = 0
        now = time.time()
        self.window_cars_left = 0
        self.window_cars_right = 0
        self.window_total_counted = 0
        self.lifetime_cars_left = 0
        self.lifetime_cars_right = 0
        self.lifetime_total_counted = 0
        self.window_started_at = now
        self.window_last_reset_at = now
        self.window_next_reset_at = now + self.counter_window_seconds
        self.count_last_changed_at = None
        self.active_tracks = 0
        self.latest_crossing_event = None
        self.line_a_crossings_seen = 0
        self.line_b_crossings_seen = 0
        self.tracks_waiting_for_second_line = 0
        self.track_id_switches = 0
        self.counted_classes = line_counter.counted_classes if line_counter else []
        initial_counter_state = line_counter.to_status() if line_counter else {}
        self.line_a = initial_counter_state.get("line_a")
        self.line_b = initial_counter_state.get("line_b")
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
            f"{self.actual_width}x{self.actual_height} at {self.actual_camera_fps} FPS; "
            f"model: {self.active_model} ({self.model_backend}); "
            "inference target: "
            f"{self.inference_width}x{self.inference_height}; "
            f"stream max: {self.max_stream_fps} FPS",
            flush=True,
        )

        self.error = None
        self.running = True
        frames = 0
        encoded_frames = 0
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
            frame_number += 1
            now = time.monotonic()
            self._store_frame(frame_number, now, frame)

            if self.detection_enabled and frame_number % self.detection_run_every_n_frames == 0:
                self._start_detection(frame_number, frame.copy())

            if next_encode_time == 0.0:
                next_encode_time = now

            if now >= next_encode_time:
                encoded_frames += self._cache_delayed_jpeg_frame(now)
                next_encode_time += stream_interval
                if next_encode_time < now:
                    next_encode_time = now + stream_interval

            frames += 1
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                with self._detection_lock:
                    completed_detections = self._completed_detections
                    self._completed_detections = 0

                self.camera_fps = round(frames / elapsed, 1)
                self.stream_fps = round(encoded_frames / elapsed, 1)
                self.detection_fps = round(completed_detections / elapsed, 1)
                self.fps = self.stream_fps
                frames = 0
                encoded_frames = 0
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

    def _store_frame(self, frame_id, timestamp, frame):
        with self._lock:
            self._frame = frame
            self.latest_frame_id = frame_id
            self._frame_buffer.append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "raw_frame": frame.copy(),
                    "annotated_frame": None,
                    "detections": [],
                }
            )

    def _select_stream_frame(self):
        with self._lock:
            if not self._frame_buffer:
                return None

            # delay_frames is applied here. 0 streams the newest buffered frame.
            index = max(0, len(self._frame_buffer) - 1 - self.delay_frames)
            target_entry = self._frame_buffer[index]
            selected_entry = target_entry
            uses_annotated_frame = False

            if target_entry["annotated_frame"] is not None:
                uses_annotated_frame = True
            else:
                for entry in reversed(list(self._frame_buffer)[: index + 1]):
                    if entry["annotated_frame"] is not None:
                        selected_entry = entry
                        uses_annotated_frame = True
                        break

            frame = selected_entry["annotated_frame"] if uses_annotated_frame else selected_entry["raw_frame"]
            output_frame = frame.copy()
            if not uses_annotated_frame:
                output_frame = self._draw_counting_lines(output_frame)
            self.streamed_frame_id = selected_entry["frame_id"]
            self.stream_uses_annotated_frame = uses_annotated_frame
            self.boxes_drawn_count = len(selected_entry["detections"]) if uses_annotated_frame else 0
            return {
                "frame": output_frame,
                "selected_frame_id": selected_entry["frame_id"],
                "target_frame_id": target_entry["frame_id"],
                "uses_annotated_frame": uses_annotated_frame,
                "boxes_drawn_count": self.boxes_drawn_count,
            }

    def _cache_delayed_jpeg_frame(self, now):
        selected = self._select_stream_frame()
        if selected is None:
            return 0

        jpeg_frame = self._encode_frame(selected["frame"])
        if jpeg_frame is None:
            return 0

        with self._lock:
            self._jpeg_frame = jpeg_frame

        self._maybe_log_sync(now, selected)
        return 1

    def _start_detection(self, frame_id, frame):
        with self._detection_lock:
            if self._detection_busy:
                return
            self._detection_busy = True

        thread = threading.Thread(target=self._run_detection, args=(frame_id, frame), daemon=True)
        thread.start()

    def _run_detection(self, frame_id, frame):
        started_at = time.monotonic()
        original_height, original_width = frame.shape[:2]
        inference_frame = self._resize_for_inference(frame)
        try:
            inference_detections = self.detector.detect(inference_frame)
        except Exception as exc:
            self.detection_error = str(exc)
            print(f"Detection error: {exc}", flush=True)
            with self._detection_lock:
                self._detection_busy = False
            return

        elapsed_ms = (time.monotonic() - started_at) * 1000
        detections = self._scale_detections_to_original(
            inference_detections,
            inference_frame.shape[1],
            inference_frame.shape[0],
            original_width,
            original_height,
        )
        counter_state = self._update_counter(frame, detections)
        annotated_frame = self._draw_detections(frame, detections)
        annotated_frame = self._draw_counting_lines(annotated_frame)
        with self._lock:
            for entry in self._frame_buffer:
                if entry["frame_id"] == frame_id:
                    entry["detections"] = detections
                    entry["annotated_frame"] = annotated_frame
                    break

        self.inference_ms = round(elapsed_ms, 1)
        self.inference_frame_width = inference_frame.shape[1]
        self.inference_frame_height = inference_frame.shape[0]
        self.model_input_width = getattr(self.detector, "effective_input_width", self.inference_frame_width)
        self.model_input_height = getattr(self.detector, "effective_input_height", self.inference_frame_height)
        self.vehicle_detections_count = len(detections)
        self.detection_frame_id = frame_id
        self.detection_error = None
        self._apply_counter_state(counter_state)
        with self._detection_lock:
            self._completed_detections += 1
            self._detection_busy = False

    def _resize_for_inference(self, frame):
        frame_height, frame_width = frame.shape[:2]
        target_width = max(1, int(self.inference_width or frame_width))
        target_height = max(1, int(self.inference_height or frame_height))

        if target_width == frame_width and target_height == frame_height:
            return frame

        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def _scale_detections_to_original(self, detections, inference_width, inference_height, original_width, original_height):
        if inference_width == original_width and inference_height == original_height:
            return detections

        x_scale = original_width / max(1, inference_width)
        y_scale = original_height / max(1, inference_height)
        scaled = []

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            scaled_detection = dict(detection)
            scaled_detection["bbox"] = [
                max(0, min(original_width - 1, int(round(x1 * x_scale)))),
                max(0, min(original_height - 1, int(round(y1 * y_scale)))),
                max(0, min(original_width - 1, int(round(x2 * x_scale)))),
                max(0, min(original_height - 1, int(round(y2 * y_scale)))),
            ]
            scaled.append(scaled_detection)

        return scaled

    def _draw_detections(self, frame, detections):
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

    def _update_counter(self, frame, detections):
        if not self.line_counter:
            return None

        frame_height, frame_width = frame.shape[:2]
        return self.line_counter.update(detections, frame_width, frame_height, time.monotonic())

    def _apply_counter_state(self, state):
        if not state:
            return

        self._apply_counter_totals(
            state["cars_left"],
            state["cars_right"],
            state["total_counted"],
        )
        self.active_tracks = state["active_tracks"]
        self.latest_crossing_event = state["latest_crossing_event"]
        self.line_a = state["line_a"]
        self.line_b = state["line_b"]
        self.line_a_crossings_seen = state["line_a_crossings_seen"]
        self.line_b_crossings_seen = state["line_b_crossings_seen"]
        self.tracks_waiting_for_second_line = state["tracks_waiting_for_second_line"]
        self.track_id_switches = state["track_id_switches"]
        self.counted_classes = state["counted_classes"]

    def get_count_window_status(self):
        self._reset_count_window_if_due()
        with self._counter_window_lock:
            seconds_until_reset = max(0, int(round(self.window_next_reset_at - time.time())))
            return {
                "cars_left": self.window_cars_left,
                "cars_right": self.window_cars_right,
                "total_counted": self.window_total_counted,
                "window_cars_left": self.window_cars_left,
                "window_cars_right": self.window_cars_right,
                "window_total_counted": self.window_total_counted,
                "lifetime_cars_left": self.lifetime_cars_left,
                "lifetime_cars_right": self.lifetime_cars_right,
                "lifetime_total_counted": self.lifetime_total_counted,
                "window_started_at": self.window_started_at,
                "window_started_at_text": self._format_timestamp(self.window_started_at),
                "window_last_reset_at": self.window_last_reset_at,
                "window_last_reset_at_text": self._format_timestamp(self.window_last_reset_at),
                "window_next_reset_at": self.window_next_reset_at,
                "window_next_reset_at_text": self._format_timestamp(self.window_next_reset_at),
                "seconds_until_window_reset": seconds_until_reset,
                "count_last_changed_at": self.count_last_changed_at,
                "count_last_changed_at_text": self._format_timestamp(self.count_last_changed_at),
                "counter_window_seconds": self.counter_window_seconds,
            }

    def _apply_counter_totals(self, lifetime_left, lifetime_right, lifetime_total):
        self._reset_count_window_if_due()
        with self._counter_window_lock:
            left_delta = max(0, lifetime_left - self.lifetime_cars_left)
            right_delta = max(0, lifetime_right - self.lifetime_cars_right)
            total_delta = max(0, lifetime_total - self.lifetime_total_counted)

            if left_delta or right_delta or total_delta:
                self.window_cars_left += left_delta
                self.window_cars_right += right_delta
                self.window_total_counted += total_delta
                self.count_last_changed_at = time.time()

            self.lifetime_cars_left = lifetime_left
            self.lifetime_cars_right = lifetime_right
            self.lifetime_total_counted = lifetime_total
            self.cars_left = self.window_cars_left
            self.cars_right = self.window_cars_right
            self.total_counted = self.window_total_counted

    def _reset_count_window_if_due(self):
        now = time.time()
        with self._counter_window_lock:
            if now < self.window_next_reset_at:
                return

            missed_windows = int((now - self.window_next_reset_at) // self.counter_window_seconds) + 1
            reset_at = self.window_next_reset_at + (missed_windows - 1) * self.counter_window_seconds
            self.window_last_reset_at = reset_at
            self.window_started_at = reset_at
            self.window_next_reset_at = reset_at + self.counter_window_seconds

            self.window_cars_left = 0
            self.window_cars_right = 0
            self.window_total_counted = 0
            self.cars_left = 0
            self.cars_right = 0
            self.total_counted = 0

    def _format_timestamp(self, timestamp):
        if not timestamp:
            return None

        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def _draw_counting_lines(self, frame):
        if not self.line_counter:
            return frame

        return self.line_counter.draw(frame)

    def _maybe_log_sync(self, now, selected):
        if now - self._last_debug_log_time < 1.0:
            return

        self._last_debug_log_time = now
        print(
            "Stream sync: "
            f"latest_frame_id={self.latest_frame_id} "
            f"detection_frame_id={self.detection_frame_id} "
            f"target_frame_id={selected['target_frame_id']} "
            f"streamed_frame_id={selected['selected_frame_id']} "
            f"delay_frames={self.delay_frames} "
            f"stream_uses_annotated_frame={selected['uses_annotated_frame']} "
            f"model={self.active_model} "
            f"detections={self.vehicle_detections_count} "
            f"boxes_drawn={selected['boxes_drawn_count']} "
            f"camera_fps={self.camera_fps} "
            f"stream_fps={self.stream_fps} "
            f"inference={self.inference_frame_width}x{self.inference_frame_height} "
            f"model_input={self.model_input_width}x{self.model_input_height} "
            f"inference_ms={self.inference_ms}",
            flush=True,
        )

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

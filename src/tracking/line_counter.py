import math

import cv2


def side_of_line(point, line_p1, line_p2):
    px, py = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _segments_intersect(p1, p2, q1, q2):
    """Return true when movement segment p1->p2 crosses counting segment q1->q2."""
    p1_side = side_of_line(p1, q1, q2)
    p2_side = side_of_line(p2, q1, q2)
    q1_side = side_of_line(q1, p1, p2)
    q2_side = side_of_line(q2, p1, p2)

    return p1_side * p2_side < 0 and q1_side * q2_side < 0


class LineCounter:
    """Small centroid tracker plus diagonal A/B line sequence counter."""

    def __init__(self, config, vehicle_classes):
        self.enabled = bool(config.get("enabled", False))
        self.draw_lines = bool(config.get("draw_lines", True))
        self.line_a_config = config.get("line_a", {})
        self.line_b_config = config.get("line_b", {})
        self.sequence_a_then_b = config.get("sequence_a_then_b", "left")
        self.sequence_b_then_a = config.get("sequence_b_then_a", "right")
        self.max_track_age_seconds = float(config.get("max_track_age_seconds", 3.0))
        self.vehicle_classes = set(vehicle_classes or ["car", "truck", "bus", "motorcycle"])

        self.cars_left = 0
        self.cars_right = 0
        self.total_counted = 0
        self.latest_crossing_event = None
        self._tracks = {}
        self._next_track_id = 1
        self._last_frame_size = None

    def update(self, detections, frame_width, frame_height, now):
        self._last_frame_size = (frame_width, frame_height)
        if not self.enabled:
            return self.to_status(frame_width, frame_height)

        vehicle_detections = [
            detection for detection in detections
            if detection.get("class_name") in self.vehicle_classes
        ]
        self._remove_stale_tracks(now)

        unmatched_track_ids = set(self._tracks.keys())
        for detection in vehicle_detections:
            center = self._detection_center(detection)
            track_id = self._match_track(center, unmatched_track_ids, frame_width, frame_height)
            if track_id is None:
                track_id = self._create_track(center, now)
            else:
                unmatched_track_ids.discard(track_id)

            self._update_track(track_id, center, now, frame_width, frame_height)

        return self.to_status(frame_width, frame_height)

    def draw(self, frame):
        if not self.draw_lines:
            return frame

        height, width = frame.shape[:2]
        for line_config in (self.line_a_config, self.line_b_config):
            name = line_config.get("name", "")
            p1, p2 = self._line_pixels(line_config, width, height)
            color = tuple(int(value) for value in line_config.get("color", [0, 0, 255]))
            cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)
            label_x = min(width - 30, max(8, p1[0] + 8))
            label_y = min(height - 8, max(24, p1[1] - 8))
            cv2.putText(
                frame,
                name,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                3,
                cv2.LINE_AA,
            )

        return frame

    def to_status(self, frame_width=None, frame_height=None):
        if frame_width is None or frame_height is None:
            frame_width, frame_height = self._last_frame_size or (0, 0)

        return {
            "cars_left": self.cars_left,
            "cars_right": self.cars_right,
            "total_counted": self.total_counted,
            "active_tracks": len(self._tracks),
            "latest_crossing_event": self.latest_crossing_event,
            "line_a": self._line_status(self.line_a_config, frame_width, frame_height),
            "line_b": self._line_status(self.line_b_config, frame_width, frame_height),
        }

    def _match_track(self, center, candidate_track_ids, frame_width, frame_height):
        best_track_id = None
        best_distance = None
        max_distance = max(48.0, math.hypot(frame_width, frame_height) * 0.08)

        for track_id in candidate_track_ids:
            track = self._tracks[track_id]
            distance = math.dist(center, track["last_center"])
            if distance <= max_distance and (best_distance is None or distance < best_distance):
                best_track_id = track_id
                best_distance = distance

        return best_track_id

    def _create_track(self, center, now):
        track_id = self._next_track_id
        self._next_track_id += 1
        self._tracks[track_id] = {
            "first_line": None,
            "counted": False,
            "last_seen": now,
            "last_center": center,
        }
        return track_id

    def _update_track(self, track_id, center, now, frame_width, frame_height):
        track = self._tracks[track_id]
        previous_center = track["last_center"]
        crossed_lines = self._crossed_lines(previous_center, center, frame_width, frame_height)

        for line_name in crossed_lines:
            self._record_line_crossing(track_id, track, line_name)

        track["last_center"] = center
        track["last_seen"] = now

    def _record_line_crossing(self, track_id, track, line_name):
        if track["counted"]:
            return

        if track["first_line"] is None:
            track["first_line"] = line_name
            return

        if track["first_line"] == line_name:
            return

        sequence = f"{track['first_line']}_then_{line_name}"
        direction = None
        if sequence == "A_then_B":
            direction = self.sequence_a_then_b
        elif sequence == "B_then_A":
            direction = self.sequence_b_then_a

        if direction == "left":
            self.cars_left += 1
        elif direction == "right":
            self.cars_right += 1
        else:
            return

        track["counted"] = True
        self.total_counted = self.cars_left + self.cars_right
        self.latest_crossing_event = f"Track {track_id} crossed {track['first_line']} then {line_name} -> {direction}"

    def _crossed_lines(self, previous_center, current_center, frame_width, frame_height):
        crossed = []
        for line_name, line_config in (("A", self.line_a_config), ("B", self.line_b_config)):
            line_p1, line_p2 = self._line_pixels(line_config, frame_width, frame_height)
            previous_side = side_of_line(previous_center, line_p1, line_p2)
            current_side = side_of_line(current_center, line_p1, line_p2)
            if previous_side * current_side < 0 and _segments_intersect(
                previous_center,
                current_center,
                line_p1,
                line_p2,
            ):
                crossed.append(line_name)

        return crossed

    def _remove_stale_tracks(self, now):
        stale_ids = [
            track_id for track_id, track in self._tracks.items()
            if now - track["last_seen"] > self.max_track_age_seconds
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]

    def _detection_center(self, detection):
        x1, y1, x2, y2 = detection["bbox"]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _line_pixels(self, line_config, frame_width, frame_height):
        p1_norm = line_config.get("p1_norm", [0.0, 0.0])
        p2_norm = line_config.get("p2_norm", [0.0, 0.0])
        return (
            (int(p1_norm[0] * frame_width), int(p1_norm[1] * frame_height)),
            (int(p2_norm[0] * frame_width), int(p2_norm[1] * frame_height)),
        )

    def _line_status(self, line_config, frame_width, frame_height):
        p1, p2 = self._line_pixels(line_config, frame_width, frame_height)
        return {
            "name": line_config.get("name"),
            "p1_norm": line_config.get("p1_norm"),
            "p2_norm": line_config.get("p2_norm"),
            "p1": list(p1),
            "p2": list(p2),
        }

import math

import cv2


def side_of_line(point, line_p1, line_p2):
    px, py = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _orientation(a, b, c):
    return side_of_line(c, a, b)


def _point_on_segment(point, seg_a, seg_b):
    px, py = point
    ax, ay = seg_a
    bx, by = seg_b
    return (
        min(ax, bx) <= px <= max(ax, bx)
        and min(ay, by) <= py <= max(ay, by)
        and abs(_orientation(seg_a, seg_b, point)) < 1e-6
    )


def segments_intersect(p1, p2, q1, q2):
    """Return true when movement segment p1->p2 crosses counting segment q1->q2."""
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    return (
        _point_on_segment(q1, p1, p2)
        or _point_on_segment(q2, p1, p2)
        or _point_on_segment(p1, q1, q2)
        or _point_on_segment(p2, q1, q2)
    )


def _intersection_t(p1, p2, q1, q2):
    """Return where q1->q2 intersects p1->p2, as t from 0.0 to 1.0."""
    px, py = p1
    rx, ry = p2[0] - p1[0], p2[1] - p1[1]
    qx, qy = q1
    sx, sy = q2[0] - q1[0], q2[1] - q1[1]
    denominator = rx * sy - ry * sx

    if abs(denominator) < 1e-6:
        return 0.0

    return ((qx - px) * sy - (qy - py) * sx) / denominator


class LineCounter:
    """Small centroid tracker plus diagonal A/B line sequence counter."""

    def __init__(self, config, vehicle_classes, tracking_config=None, debug_config=None):
        tracking_config = tracking_config or {}
        debug_config = debug_config or {}

        self.enabled = bool(config.get("enabled", False))
        self.draw_lines = bool(config.get("draw_lines", True))
        self.label_offset_px = config.get("label_offset_px", [-28, -14])
        self.line_a_config = config.get("line_a", {})
        self.line_b_config = config.get("line_b", {})
        self.sequence_a_then_b = config.get("sequence_a_then_b", "left")
        self.sequence_b_then_a = config.get("sequence_b_then_a", "right")
        self.max_track_age_seconds = float(
            tracking_config.get("max_age_seconds", config.get("max_track_age_seconds", 5.0))
        )
        self.max_distance_px = float(tracking_config.get("max_distance_px", 160))
        self.vehicle_classes = set(vehicle_classes or ["car", "truck", "bus", "motorcycle"])

        self.draw_track_centers = bool(debug_config.get("draw_track_centers", False))
        self.draw_track_ids = bool(debug_config.get("draw_track_ids", False))
        self.draw_movement_segments = bool(debug_config.get("draw_movement_segments", False))

        self.cars_left = 0
        self.cars_right = 0
        self.total_counted = 0
        self.line_a_crossings_seen = 0
        self.line_b_crossings_seen = 0
        self.track_id_switches = 0
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
            track_id = self._match_track(center, unmatched_track_ids)
            if track_id is None:
                self._record_possible_id_switch(center)
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
            self._draw_line_label(frame, name, p1, color)

        self._draw_tracks(frame)
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
            "line_a_crossings_seen": self.line_a_crossings_seen,
            "line_b_crossings_seen": self.line_b_crossings_seen,
            "tracks_waiting_for_second_line": self._tracks_waiting_for_second_line(),
            "track_id_switches": self.track_id_switches,
        }

    def _match_track(self, center, candidate_track_ids):
        best_track_id = None
        best_distance = None

        for track_id in candidate_track_ids:
            track = self._tracks[track_id]
            distance = math.dist(center, track["last_center"])
            if distance <= self.max_distance_px and (best_distance is None or distance < best_distance):
                best_track_id = track_id
                best_distance = distance

        return best_track_id

    def _record_possible_id_switch(self, center):
        for track in self._tracks.values():
            if math.dist(center, track["last_center"]) <= self.max_distance_px * 1.6:
                self.track_id_switches += 1
                return

    def _create_track(self, center, now):
        track_id = self._next_track_id
        self._next_track_id += 1
        self._tracks[track_id] = {
            "first_line": None,
            "counted": False,
            "last_seen": now,
            "previous_center": center,
            "last_center": center,
            "crossed_lines": [],
        }
        return track_id

    def _update_track(self, track_id, center, now, frame_width, frame_height):
        track = self._tracks[track_id]
        previous_center = track["last_center"]
        crossed_lines = self._crossed_lines(previous_center, center, frame_width, frame_height)

        for line_name in crossed_lines:
            self._record_line_crossing(track_id, track, line_name)

        track["previous_center"] = previous_center
        track["last_center"] = center
        track["last_seen"] = now

    def _record_line_crossing(self, track_id, track, line_name):
        if track["counted"]:
            return

        if track["crossed_lines"] and track["crossed_lines"][-1] == line_name:
            return

        track["crossed_lines"].append(line_name)
        if line_name == "A":
            self.line_a_crossings_seen += 1
        elif line_name == "B":
            self.line_b_crossings_seen += 1

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
            if segments_intersect(previous_center, current_center, line_p1, line_p2):
                crossed.append(
                    (
                        _intersection_t(previous_center, current_center, line_p1, line_p2),
                        line_name,
                    )
                )

        return [line_name for _, line_name in sorted(crossed)]

    def _remove_stale_tracks(self, now):
        stale_ids = [
            track_id for track_id, track in self._tracks.items()
            if now - track["last_seen"] > self.max_track_age_seconds
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]

    def _tracks_waiting_for_second_line(self):
        return sum(
            1 for track in self._tracks.values()
            if track["first_line"] is not None and not track["counted"]
        )

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

    def _draw_line_label(self, frame, name, line_start, color):
        if not name:
            return

        height, width = frame.shape[:2]
        offset_x, offset_y = self.label_offset_px
        label_x = min(width - 30, max(8, line_start[0] + int(offset_x)))
        label_y = min(height - 8, max(24, line_start[1] + int(offset_y)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.85
        thickness = 2
        text_size, baseline = cv2.getTextSize(name, font, scale, thickness)
        text_w, text_h = text_size
        top_left = (label_x - 6, label_y - text_h - 6)
        bottom_right = (label_x + text_w + 6, label_y + baseline + 6)

        cv2.rectangle(frame, top_left, bottom_right, (8, 11, 16), -1)
        cv2.rectangle(frame, top_left, bottom_right, color, 1)
        cv2.putText(frame, name, (label_x, label_y), font, scale, color, thickness, cv2.LINE_AA)

    def _draw_tracks(self, frame):
        if not (self.draw_track_centers or self.draw_track_ids or self.draw_movement_segments):
            return

        for track_id, track in self._tracks.items():
            center = tuple(int(value) for value in track["last_center"])
            previous = tuple(int(value) for value in track["previous_center"])
            if self.draw_movement_segments:
                cv2.line(frame, previous, center, (255, 214, 102), 2, cv2.LINE_AA)
            if self.draw_track_centers:
                cv2.circle(frame, center, 4, (255, 214, 102), -1, cv2.LINE_AA)
            if self.draw_track_ids:
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (center[0] + 8, max(20, center[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 214, 102),
                    2,
                    cv2.LINE_AA,
                )

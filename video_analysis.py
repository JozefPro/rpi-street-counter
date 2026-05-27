"""
Variant 2 — Car line-crossing counter
Detects and tracks vehicles with YOLOv8 + ByteTrack, counts how many
cross a user-defined line, and visualises bounding-box state:
  Yellow  = newly appeared vehicle  (first few frames)
  Green   = normally tracked
  Orange  = currently crossing the line
  Blue    = has already crossed
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os

VIDEO_PATH = "traffic_video.avi"  # path to input video
OUTPUT_PATH = "output_v2.mp4"  # path to output video
MODEL = "yolov8n.pt"  # YOLO model weights
CONF = 0.2  # detection confidence threshold
# Set to (x1, y1, x2, y2) to skip interactive line selection, or None to draw it
LINE = None  # e.g. (0, 540, 1920, 540)

# COCO class IDs for vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# BGR colours
COLOR_NEW = (0, 255, 255)  # yellow
COLOR_TRACKED = (0, 200, 0)  # green
COLOR_CROSSING = (0, 128, 255)  # orange
COLOR_CROSSED = (220, 60, 0)  # blue
COLOR_LINE = (0, 0, 255)  # red

NEW_TRACK_FRAMES = 8  # frames to display a car as "new"
CROSSING_SHOW_FRAMES = 5  # frames to keep the "crossing" highlight


def side_of_line(point, p1, p2):
    """Sign of the cross-product — tells which side of line p1→p2 a point is on."""
    return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])


def select_counting_line(frame):
    """Interactive: click two points to define the counting line, then press Enter."""
    points = []
    display = [frame.copy()]  # list so the callback can mutate it in-place
    title = "Select Counting Line (click 2 pts, Enter=confirm, Esc=cancel)"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            d = frame.copy()
            for p in points:
                cv2.circle(d, p, 7, COLOR_LINE, -1)
            if len(points) == 2:
                cv2.line(d, points[0], points[1], COLOR_LINE, 2)
            display[0] = d  # mutate the list — main loop sees the update

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)

    while True:
        cv2.imshow(title, display[0])  # always refresh so clicks appear immediately
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(points) == 2:  # Enter — confirm
            break
        if key == 27:  # Esc — abort
            points.clear()
            break

    cv2.destroyWindow(title)
    return tuple(points) if len(points) == 2 else None


def draw_semi_transparent_rect(img, pt1, pt2, color=(0, 0, 0), alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def main():
    model = YOLO(MODEL)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Select counting line ──────────────────────────────────────────────────
    if LINE is not None:
        line_p1, line_p2 = (LINE[0], LINE[1]), (LINE[2], LINE[3])
    else:
        ret, first = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from video")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        result = select_counting_line(first)
        if result is None:
            print("No line selected — using horizontal centre line")
            line_p1, line_p2 = (0, H // 2), (W, H // 2)
        else:
            line_p1, line_p2 = result

    print(f"Counting line: {line_p1} → {line_p2}")

    # ── Output video writer ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # ── Tracking state ────────────────────────────────────────────────────────
    track_first_frame = {}  # id → frame first seen
    track_side = {}  # id → last side value (float, sign matters)
    track_cross_frame = {}  # id → frame number of last crossing
    track_history = defaultdict(list)  # id → list of (cx, cy)
    crossed_ids = set()
    crossed_count = 0
    frame_idx = 0

    print("Processing…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_CLASSES.keys()),
            conf=CONF,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        vis = frame.copy()

        # ── Draw counting line ────────────────────────────────────────────────
        cv2.line(vis, line_p1, line_p2, COLOR_LINE, 3)
        # Small arrow-head hint at right end of line
        mid = ((line_p1[0] + line_p2[0]) // 2, (line_p1[1] + line_p2[1]) // 2)
        cv2.putText(
            vis,
            "COUNTING LINE",
            (mid[0] + 6, mid[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_LINE,
            2,
            cv2.LINE_AA,
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), tid, cls, conf in zip(boxes, ids, classes, confs):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                center = (cx, cy)

                if tid not in track_first_frame:
                    track_first_frame[tid] = frame_idx

                track_history[tid].append(center)
                if len(track_history[tid]) > 40:
                    track_history[tid].pop(0)

                cur_side = side_of_line(center, line_p1, line_p2)

                if tid in track_side:
                    prev_side = track_side[tid]
                    # Sign flip → vehicle crossed the line (segment bounds check)
                    if (prev_side > 0) != (cur_side > 0):
                        min_x = min(line_p1[0], line_p2[0])
                        max_x = max(line_p1[0], line_p2[0])
                        min_y = min(line_p1[1], line_p2[1])
                        max_y = max(line_p1[1], line_p2[1])
                        within_segment = min_x <= cx <= max_x and min_y <= cy <= max_y
                        if within_segment:
                            if tid not in crossed_ids:
                                crossed_ids.add(tid)
                                crossed_count += 1
                            track_cross_frame[tid] = frame_idx

                track_side[tid] = cur_side

                # ── Determine visual state ────────────────────────────────────
                is_new = (frame_idx - track_first_frame[tid]) < NEW_TRACK_FRAMES
                is_crossing = (
                    tid in track_cross_frame
                    and frame_idx - track_cross_frame[tid] < CROSSING_SHOW_FRAMES
                )
                has_crossed = tid in crossed_ids

                if is_crossing:
                    color, state_label = COLOR_CROSSING, "CROSSING!"
                elif is_new:
                    color, state_label = COLOR_NEW, "NEW"
                elif has_crossed:
                    color, state_label = COLOR_CROSSED, "CROSSED"
                else:
                    color, state_label = COLOR_TRACKED, VEHICLE_CLASSES.get(
                        cls, "vehicle"
                    )

                thick = 3 if is_crossing else 2

                # Bounding box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)

                # Movement trail
                pts = track_history[tid]
                for i in range(1, len(pts)):
                    cv2.line(vis, pts[i - 1], pts[i], color, 1)

                # Label
                label = f"ID {tid}  {state_label}  {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(vis, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
                cv2.putText(
                    vis,
                    label,
                    (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.circle(vis, center, 4, color, -1)

        # ── HUD (top-left stats) ──────────────────────────────────────────────
        draw_semi_transparent_rect(vis, (8, 8), (320, 115))
        cv2.putText(
            vis,
            f"Crossed line: {crossed_count}",
            (18, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"Total tracked: {len(track_first_frame)}",
            (18, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"Frame: {frame_idx} / {total_frames}",
            (18, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # ── Legend (bottom-left) ──────────────────────────────────────────────
        legend_items = [
            (COLOR_NEW, "New vehicle"),
            (COLOR_TRACKED, "Tracked"),
            (COLOR_CROSSING, "Crossing line"),
            (COLOR_CROSSED, "Has crossed"),
        ]
        ly = H - len(legend_items) * 24 - 16
        draw_semi_transparent_rect(vis, (8, ly - 4), (200, H - 8))
        for i, (c, txt) in enumerate(legend_items):
            y = ly + 14 + i * 24
            cv2.rectangle(vis, (15, y - 9), (34, y + 7), c, -1)
            cv2.putText(
                vis,
                txt,
                (40, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        out.write(vis)

        if frame_idx % 50 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(
                f"  [{pct:5.1f}%]  frame {frame_idx}/{total_frames}"
                f"  |  crossed: {crossed_count}"
            )

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n=== Final Results ===")
    print(f"Vehicles that crossed the line : {crossed_count}")
    print(f"Unique vehicles tracked        : {len(track_first_frame)}")
    print(f"Output saved to                : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

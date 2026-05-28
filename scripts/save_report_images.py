#!/usr/bin/env python3
"""Create report screenshots for chapter 6 system implementation sections."""

from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import create_detector


DEFAULT_CONFIG = {
    "camera": {"index": 0, "width": 1280, "height": 720, "fps": 30},
    "stream": {"jpeg_quality": 70},
    "detection": {
        "enabled": True,
        "model": "yolo_nano",
        "confidence_threshold": 0.35,
        "classes": ["car", "truck", "bus", "motorcycle"],
        "input_size": 640,
    },
    "counting": {
        "line_a": {"name": "A", "p1_norm": [0.14, 0.66], "p2_norm": [0.69, 0.28], "color": [0, 0, 255]},
        "line_b": {"name": "B", "p1_norm": [0.42, 0.98], "p2_norm": [0.94, 0.47], "color": [0, 0, 255]},
    },
    "models": {},
}


def load_config(path: Path) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        for section, values in loaded.items():
            if isinstance(values, dict) and isinstance(config.get(section), dict):
                config[section] = {**config[section], **values}
            else:
                config[section] = values
    return config


def capture_frame(config: dict, warmup_frames: int) -> tuple[bool, object]:
    camera_config = config["camera"]
    cap = cv2.VideoCapture(camera_config.get("index", 0))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config.get("width", 1280)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config.get("height", 720)))
    cap.set(cv2.CAP_PROP_FPS, int(camera_config.get("fps", 30)))

    if not cap.isOpened():
        return False, make_fallback_frame("Camera unavailable")

    frame = None
    for _ in range(max(1, warmup_frames)):
        ok, candidate = cap.read()
        if ok:
            frame = candidate
        time.sleep(0.03)

    cap.release()
    if frame is None:
        return False, make_fallback_frame("Camera frame unavailable")

    return True, frame


def make_fallback_frame(message: str):
    frame = cv2.imread("saved_images/6_1_raw_camera_stream.jpg")
    if frame is not None:
        return frame

    import numpy as np

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(frame, message, (70, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 230, 230), 2, cv2.LINE_AA)
    return frame


def save_image(path: Path, frame, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    if not cv2.imwrite(str(path), frame, params):
        raise RuntimeError(f"Could not save image: {path}")


def draw_detections(frame, detections, show_only_configured: bool = False, accepted_classes: set[str] | None = None):
    image = frame.copy()
    for detection in detections:
        class_name = detection["class_name"]
        if show_only_configured and accepted_classes is not None and class_name not in accepted_classes:
            continue

        x1, y1, x2, y2 = detection["bbox"]
        confidence = detection["confidence"]
        label = f"{class_name} {confidence:.2f}"
        color = (57, 208, 163)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = text_size
        top = max(0, y1 - text_h - baseline - 8)
        cv2.rectangle(image, (x1, top), (x1 + text_w + 8, top + text_h + baseline + 8), (8, 11, 16), -1)
        cv2.putText(image, label, (x1 + 4, top + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return image


def line_pixels(line_config: dict, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    p1_norm = line_config["p1_norm"]
    p2_norm = line_config["p2_norm"]
    return (
        (int(p1_norm[0] * width), int(p1_norm[1] * height)),
        (int(p2_norm[0] * width), int(p2_norm[1] * height)),
    )


def draw_counting_lines(frame, config: dict, include_labels: bool) -> object:
    image = frame.copy()
    height, width = image.shape[:2]
    for line_key in ("line_a", "line_b"):
        line_config = config["counting"][line_key]
        p1, p2 = line_pixels(line_config, width, height)
        color = tuple(int(value) for value in line_config.get("color", [0, 0, 255]))
        cv2.line(image, p1, p2, color, 4, cv2.LINE_AA)
        if include_labels:
            cv2.putText(image, line_config["name"], (p1[0] + 8, max(28, p1[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3, cv2.LINE_AA)
    return image


def draw_direction_logic(frame, config: dict) -> object:
    image = draw_counting_lines(frame, config, include_labels=True)
    height, width = image.shape[:2]
    line_a_p1, line_a_p2 = line_pixels(config["counting"]["line_a"], width, height)
    line_b_p1, line_b_p2 = line_pixels(config["counting"]["line_b"], width, height)
    a_mid = ((line_a_p1[0] + line_a_p2[0]) // 2, (line_a_p1[1] + line_a_p2[1]) // 2)
    b_mid = ((line_b_p1[0] + line_b_p2[0]) // 2, (line_b_p1[1] + line_b_p2[1]) // 2)

    cv2.arrowedLine(image, a_mid, b_mid, (255, 214, 102), 5, cv2.LINE_AA, tipLength=0.08)
    cv2.arrowedLine(image, b_mid, a_mid, (57, 208, 163), 4, cv2.LINE_AA, tipLength=0.08)
    draw_label(image, "A then B: left", (min(a_mid[0], b_mid[0]) + 20, min(a_mid[1], b_mid[1]) - 30), (255, 214, 102))
    draw_label(image, "B then A: right", (min(a_mid[0], b_mid[0]) + 20, min(a_mid[1], b_mid[1]) + 24), (57, 208, 163))
    return image


def draw_label(image, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    text_w, text_h = text_size
    cv2.rectangle(image, (x - 8, y - text_h - 8), (x + text_w + 8, y + baseline + 8), (8, 11, 16), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Save chapter 6 report images.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="saved_images")
    parser.add_argument("--warmup-frames", type=int, default=10)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = Path(args.output_dir)
    quality = int(config.get("stream", {}).get("jpeg_quality", 80))

    camera_ok, frame = capture_frame(config, args.warmup_frames)
    save_image(output_dir / "6_1_raw_camera_stream.jpg", frame, quality)

    detections = []
    if camera_ok and config.get("detection", {}).get("enabled", False):
        detector = create_detector(config)
        detections = detector.detect(frame)

    accepted_classes = set(config.get("detection", {}).get("classes", []))
    save_image(output_dir / "6_2_object_detection.jpg", draw_detections(frame, detections), quality)
    save_image(
        output_dir / "6_3_object_filtering.jpg",
        draw_detections(frame, detections, show_only_configured=True, accepted_classes=accepted_classes),
        quality,
    )
    save_image(output_dir / "6_4_counting_lines.jpg", draw_counting_lines(frame, config, include_labels=False), quality)
    save_image(output_dir / "6_5_direction_detection.jpg", draw_direction_logic(frame, config), quality)

    print(f"Camera frame captured: {camera_ok}")
    print(f"Detections drawn: {len(detections)}")
    for image_path in sorted(output_dir.glob("6_*.jpg")):
        print(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

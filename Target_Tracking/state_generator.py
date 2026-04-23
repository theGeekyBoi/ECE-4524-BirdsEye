"""
State vector generator for the BirdsEye project.

Combines the physical car detector (red body + white front marker) and the
HSV-thresholded target detector into a single 22-dimensional normalized float
state vector.

Coordinate space is fixed at 900x600 to match the logical map used by
TestRig.py (SCREEN_WIDTH=900, SCREEN_HEIGHT=600). The captured camera frame
is always resized to that size before detection runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import cv2
import numpy as np

from car_detector import (
    _build_red_mask,
    _configure_capture,
    _open_camera,
    detect_physical_car,
)
from target_detector import detect_target


MAP_WIDTH = 900
MAP_HEIGHT = 600
MAP_DIAGONAL = math.hypot(MAP_WIDTH, MAP_HEIGHT)
DEFAULT_CAMERA_INDEX = 1
STATE_VECTOR_SIZE = 22
DEBUG_WINDOW_NAME = "State Generator Debug"


# Ordered (index, label) pairs that describe every slot in the state vector.
# The order here must exactly match the order produced by _assemble_vector.
STATE_VECTOR_LABELS: list[tuple[int, str]] = [
    (0, "fr_x"),
    (1, "fr_y"),
    (2, "fl_x"),
    (3, "fl_y"),
    (4, "br_x"),
    (5, "br_y"),
    (6, "bl_x"),
    (7, "bl_y"),
    (8, "sin_heading"),
    (9, "cos_heading"),
    (10, "forward_x"),
    (11, "forward_y"),
    (12, "target_x"),
    (13, "target_y"),
    (14, "target_radius"),
    (15, "dist_to_target"),
    (16, "delta_x"),
    (17, "delta_y"),
    (18, "dist_left"),
    (19, "dist_right"),
    (20, "dist_top"),
    (21, "dist_bottom"),
]


def _assemble_vector(car: dict, tgt: dict) -> np.ndarray:
    """
    Build the 22-dim normalized state vector from detector outputs.
    """
    corners = car["car_corners"]
    fr_x, fr_y = corners["front_right"]
    fl_x, fl_y = corners["front_left"]
    br_x, br_y = corners["back_right"]
    bl_x, bl_y = corners["back_left"]

    heading_deg = float(car["car_direction"]["angle_degrees"])
    heading_rad = math.radians(heading_deg)
    sin_h = math.sin(heading_rad)
    cos_h = math.cos(heading_rad)

    forward_x, forward_y = car["car_direction"]["forward_vector"]

    car_cx, car_cy = car["car_center"]

    tgt_x, tgt_y = tgt["center"]
    tgt_radius = float(tgt["radius"])

    dx = tgt_x - car_cx
    dy = tgt_y - car_cy
    distance = math.hypot(dx, dy)

    values = [
        fr_x / MAP_WIDTH,
        fr_y / MAP_HEIGHT,
        fl_x / MAP_WIDTH,
        fl_y / MAP_HEIGHT,
        br_x / MAP_WIDTH,
        br_y / MAP_HEIGHT,
        bl_x / MAP_WIDTH,
        bl_y / MAP_HEIGHT,
        sin_h,
        cos_h,
        float(forward_x),
        float(forward_y),
        tgt_x / MAP_WIDTH,
        tgt_y / MAP_HEIGHT,
        tgt_radius / MAP_WIDTH,
        distance / MAP_DIAGONAL,
        dx / MAP_DIAGONAL,
        dy / MAP_DIAGONAL,
        car_cx / MAP_WIDTH,
        (MAP_WIDTH - car_cx) / MAP_WIDTH,
        car_cy / MAP_HEIGHT,
        (MAP_HEIGHT - car_cy) / MAP_HEIGHT,
    ]

    vector = np.asarray(values, dtype=np.float32)
    assert vector.shape == (STATE_VECTOR_SIZE,), (
        f"expected shape ({STATE_VECTOR_SIZE},), got {vector.shape}"
    )
    return vector


def _resize_to_map(frame_bgr: np.ndarray) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    if width == MAP_WIDTH and height == MAP_HEIGHT:
        return frame_bgr
    return cv2.resize(
        frame_bgr, (MAP_WIDTH, MAP_HEIGHT), interpolation=cv2.INTER_AREA
    )


def build_state(frame_bgr: np.ndarray) -> dict:
    """
    Resize the frame to 900x600, run both detectors, and return both the
    raw detections and the assembled state vector.
    """
    resized = _resize_to_map(frame_bgr)
    car = detect_physical_car(resized)
    tgt = detect_target(resized)
    vector = _assemble_vector(car, tgt)
    return {
        "resized_frame": resized,
        "car": car,
        "target": tgt,
        "state_vector": vector,
    }


def build_state_vector(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper that returns only the 22-dim state vector.
    """
    return build_state(frame_bgr)["state_vector"]


def format_state_vector(vector: np.ndarray) -> str:
    """
    Produce a human-readable block listing every element of the state vector
    together with its label, plus a compact raw-array line.
    """
    if vector.shape != (STATE_VECTOR_SIZE,):
        raise ValueError(
            f"state vector must have shape ({STATE_VECTOR_SIZE},), "
            f"got {vector.shape}"
        )

    max_label_len = max(len(label) for _, label in STATE_VECTOR_LABELS)
    lines = [
        f"state_vector (shape=({STATE_VECTOR_SIZE},), dtype={vector.dtype}):"
    ]
    for idx, label in STATE_VECTOR_LABELS:
        value = float(vector[idx])
        lines.append(
            f"  [{idx:2d}]  {label:<{max_label_len}} = {value: .6f}"
        )
    raw = ", ".join(f"{float(v): .6f}" for v in vector)
    lines.append(f"raw_array: [{raw}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization helpers (used only when --display is passed).
# ---------------------------------------------------------------------------


def _draw_hud_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float = 0.55,
    text_color: tuple[int, int, int] = (255, 255, 255),
    box_color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    padding: int = 5,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    left = max(0, x - padding)
    top = max(0, y - text_h - padding)
    right = min(image.shape[1] - 1, x + text_w + padding)
    bottom = min(image.shape[0] - 1, y + baseline + padding)
    cv2.rectangle(image, (left, top), (right, bottom), box_color, -1)
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def render_debug_view(
    resized_frame: np.ndarray,
    car: dict,
    tgt: dict,
    vector: np.ndarray,
) -> np.ndarray:
    """
    Build a 900x600 BGR debug image showing car + target masks, the oriented
    car polygon, the target circle, and a heading arrow.
    """
    canvas = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)

    # Car red-body mask painted in red.
    red_mask = _build_red_mask(resized_frame)
    canvas[red_mask > 0] = (0, 0, 255)

    # Target mask painted in cyan.
    target_mask = tgt.get("mask")
    if target_mask is not None:
        canvas[target_mask > 0] = (255, 255, 0)

    # Oriented car polygon in green.
    corners = car["car_corners"]
    polygon = np.array(
        [
            corners["front_right"],
            corners["front_left"],
            corners["back_left"],
            corners["back_right"],
        ],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    # Target enclosing circle outline in white.
    tgt_center = (int(tgt["center"][0]), int(tgt["center"][1]))
    tgt_radius = max(1, int(round(tgt["radius"])))
    cv2.circle(canvas, tgt_center, tgt_radius, (255, 255, 255), 2)
    cv2.circle(canvas, tgt_center, 3, (255, 255, 255), -1)

    # Heading arrow from car center along forward vector.
    car_center = car["car_center"]
    forward = car["car_direction"]["forward_vector"]
    arrow_length = 60.0
    start = (int(round(car_center[0])), int(round(car_center[1])))
    end = (
        int(round(car_center[0] + forward[0] * arrow_length)),
        int(round(car_center[1] + forward[1] * arrow_length)),
    )
    cv2.circle(canvas, start, 4, (0, 255, 255), -1)
    cv2.arrowedLine(canvas, start, end, (255, 0, 0), 2, tipLength=0.25)

    # Front-midpoint marker for clarity.
    front_mid = car["car_direction"]["front_midpoint"]
    front_px = (int(round(front_mid[0])), int(round(front_mid[1])))
    cv2.circle(canvas, front_px, 4, (0, 128, 255), -1)

    # HUD overlay.
    heading_deg = float(car["car_direction"]["angle_degrees"])
    hud_lines = [
        f"heading={heading_deg:6.1f} deg",
        f"car=({start[0]:4d}, {start[1]:4d}) px",
        f"target=({tgt_center[0]:4d}, {tgt_center[1]:4d}) px  r={tgt['radius']:5.1f}",
        f"dist={float(vector[15]):.3f}  delta=({float(vector[16]):+.3f}, {float(vector[17]):+.3f})",
    ]
    line_y = 22
    for line in hud_lines:
        _draw_hud_text(canvas, line, (12, line_y))
        line_y += 22

    return canvas


def _render_failure_view(resized_frame: np.ndarray, message: str) -> np.ndarray:
    view = resized_frame.copy()
    _draw_hud_text(
        view,
        "detection failed",
        (12, 28),
        font_scale=0.8,
        thickness=2,
        padding=6,
    )
    _draw_hud_text(view, message, (12, 58))
    return view


def _prepare_debug_window() -> None:
    cv2.namedWindow(
        DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    )
    cv2.resizeWindow(DEBUG_WINDOW_NAME, MAP_WIDTH, MAP_HEIGHT)


# ---------------------------------------------------------------------------
# CLI plumbing.
# ---------------------------------------------------------------------------


def _camera_open_error(camera_index: int) -> RuntimeError:
    return RuntimeError(f"Could not open camera index {camera_index}.")


def _capture_single_frame(camera_index: int) -> np.ndarray:
    capture = _open_camera(camera_index)
    _configure_capture(capture, 1920, 1080)
    if not capture.isOpened():
        raise _camera_open_error(camera_index)
    try:
        # Some webcams need a couple of reads before they deliver a stable frame.
        frame = None
        for _ in range(5):
            ok, frame = capture.read()
            if ok and frame is not None:
                break
        if frame is None:
            raise RuntimeError("Failed to read a frame from the webcam.")
        return frame
    finally:
        capture.release()


def _run_once_from_frame(
    frame: np.ndarray,
    display: bool,
    save_json: str | None,
) -> None:
    result = build_state(frame)
    vector = result["state_vector"]
    print(format_state_vector(vector))

    if save_json:
        payload = {
            "labels": [label for _, label in STATE_VECTOR_LABELS],
            "values": [float(v) for v in vector],
        }
        with open(save_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved state vector JSON to: {save_json}")

    if display:
        view = render_debug_view(
            result["resized_frame"],
            result["car"],
            result["target"],
            vector,
        )
        _prepare_debug_window()
        cv2.imshow(DEBUG_WINDOW_NAME, view)
        print("Press any key in the debug window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _run_continuous(
    camera_index: int,
    interval: float,
    display: bool,
) -> None:
    capture = _open_camera(camera_index)
    _configure_capture(capture, 1920, 1080)
    if not capture.isOpened():
        raise _camera_open_error(camera_index)

    window_ready = False
    try:
        last_time = 0.0
        while True:
            now = time.perf_counter()
            if now - last_time < interval:
                if display:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read a frame from the webcam.")

            resized = _resize_to_map(frame)
            view: np.ndarray | None = None
            try:
                car = detect_physical_car(resized)
                tgt = detect_target(resized)
                vector = _assemble_vector(car, tgt)
                print(format_state_vector(vector))
                if display:
                    view = render_debug_view(resized, car, tgt, vector)
            except ValueError as exc:
                print(f"[state_generator] detection failed: {exc}")
                if display:
                    view = _render_failure_view(resized, str(exc))

            if display and view is not None:
                if not window_ready:
                    _prepare_debug_window()
                    window_ready = True
                cv2.imshow(DEBUG_WINDOW_NAME, view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            last_time = now
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the physical car detector and the HSV target detector "
            "into a single 22-dim normalized state vector."
        )
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="Webcam index for OpenCV VideoCapture (default: 1).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional image path to use instead of a live webcam frame.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Loop over webcam frames and emit a state vector per frame.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0 / 15.0,
        help="Seconds between processed frames when using --continuous.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "Open a visualization window showing the car mask, target mask, "
            "oriented car outline, and a heading arrow."
        ),
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help=(
            "Optional file path to dump the state vector as JSON "
            "(labels + values). Only applies to one-shot mode."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.input is not None:
        if not os.path.exists(args.input):
            raise RuntimeError(f"Input image does not exist: {args.input}")
        frame = cv2.imread(args.input)
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.input}")
        _run_once_from_frame(frame, display=args.display, save_json=args.save_json)
        return

    if args.continuous:
        _run_continuous(
            camera_index=args.camera_index,
            interval=args.interval,
            display=args.display,
        )
        return

    frame = _capture_single_frame(args.camera_index)
    _run_once_from_frame(frame, display=args.display, save_json=args.save_json)


if __name__ == "__main__":
    main()

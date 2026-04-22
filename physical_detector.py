"""
Physical car detector for a stationary overhead webcam.

Assumptions:
- the full camera frame is the drivable map
- the robot has a bright red top cover
- the front is marked by a small white tape marker
- the car footprint is approximately square in the image

Outputs are pixel coordinates relative to the full camera image.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time

import cv2
import numpy as np

WINDOW_NAME = "Physical Car Detection"
DEFAULT_FRAME_WIDTH = 1920
DEFAULT_FRAME_HEIGHT = 1080


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


def _draw_text_box(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float = 0.7,
    text_color: tuple[int, int, int] = (255, 255, 255),
    box_color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    padding: int = 6,
):
    """
    Draw text on top of a solid rectangle so it stays visible after preview scaling.
    """
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


def _heading_from_vector(forward_vector: np.ndarray) -> float:
    return (math.degrees(math.atan2(-forward_vector[1], forward_vector[0])) + 360.0) % 360.0


def _running_in_wsl() -> bool:
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def _camera_open_error(camera_index: int) -> RuntimeError:
    if os.name == "posix" and _running_in_wsl() and not any(
        os.path.exists(f"/dev/video{i}") for i in range(8)
    ):
        return RuntimeError(
            "Could not open camera index "
            f"{camera_index}. No /dev/video* devices are available inside WSL2. "
            "Your NexiGo webcam is visible to Windows, but it is not currently exposed "
            "to this Linux environment. The simplest fix is to run physical_detector.py "
            "from Windows Python instead of WSL, or attach the USB webcam to WSL with "
            "USB passthrough."
        )
    return RuntimeError(f"Could not open camera index {camera_index}.")


def _open_camera(camera_index: int) -> cv2.VideoCapture:
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(camera_index)
    return capture


def _configure_capture(
    capture: cv2.VideoCapture,
    frame_width: int,
    frame_height: int,
):
    if hasattr(cv2, "VideoWriter_fourcc"):
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


def _prepare_preview_window(image: np.ndarray):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, image.shape[1], image.shape[0])


def _build_red_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # The sample robot cover reads as a salmon/pink-red rather than a deep red,
    # so keep the hue windows broad and allow moderate saturation.
    lower_red_1 = np.array([0, 70, 80], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([170, 70, 80], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _build_white_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # White tape should be bright and relatively low-saturation.
    lower_white = np.array([0, 0, 170], dtype=np.uint8)
    upper_white = np.array([180, 90, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _crop_with_padding(frame: np.ndarray, bbox: tuple[int, int, int, int], padding: int):
    x0, y0, x1, y1 = bbox
    h, w = frame.shape[:2]
    left = max(0, x0 - padding)
    top = max(0, y0 - padding)
    right = min(w, x1 + padding + 1)
    bottom = min(h, y1 + padding + 1)
    cropped = frame[top:bottom, left:right]
    return cropped, left, top


def detect_red_body(frame_bgr: np.ndarray, min_area: float = 400.0) -> dict:
    """
    Detect the red car body and approximate it with a rotated rectangle.
    """
    mask = _build_red_mask(frame_bgr)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Could not find any red regions in the camera frame.")

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area:
        raise ValueError(
            f"Detected red region is too small to be the car body (area={area:.1f})."
        )

    rect = cv2.minAreaRect(contour)
    raw_corners = cv2.boxPoints(rect).astype(np.float32)
    center = np.array(rect[0], dtype=np.float32)
    x, y, w, h = cv2.boundingRect(contour)

    return {
        "mask": mask,
        "contour": contour,
        "area": area,
        "center": center,
        "raw_corners": raw_corners,
        "rect": rect,
        "bbox": (int(x), int(y), int(x + w - 1), int(y + h - 1)),
    }


def detect_white_marker_near_body(
    frame_bgr: np.ndarray,
    body_bbox: tuple[int, int, int, int],
    body_center: np.ndarray,
    min_area: float = 12.0,
    padding: int = 6,
) -> dict:
    """
    Detect a small white tape marker near the front of the red robot body.
    """
    cropped, offset_x, offset_y = _crop_with_padding(frame_bgr, body_bbox, padding)
    mask = _build_white_mask(cropped)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Could not detect a white tape marker near the car body.")

    body_center_local = body_center - np.array([offset_x, offset_y], dtype=np.float32)
    best = None
    best_score = -float("inf")

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            continue

        center_x = float(moments["m10"] / moments["m00"])
        center_y = float(moments["m01"] / moments["m00"])
        center = np.array([center_x, center_y], dtype=np.float32)

        # Favor candidate markers that are farther from the robot center.
        distance_from_center = float(np.linalg.norm(center - body_center_local))
        score = distance_from_center + 0.1 * area
        if score > best_score:
            x, y, w, h = cv2.boundingRect(contour)
            best_score = score
            best = {
                "type": "white_tape",
                "center": center + np.array([offset_x, offset_y], dtype=np.float32),
                "area": area,
                "bbox": (
                    int(x + offset_x),
                    int(y + offset_y),
                    int(x + offset_x + w - 1),
                    int(y + offset_y + h - 1),
                ),
            }

    if best is None:
        raise ValueError("White regions were found, but none looked like a usable tape marker.")

    return best
def _corners_and_heading_from_front_pair(
    raw_corners: np.ndarray,
    center: np.ndarray,
    front_indices: list[int],
) -> tuple[dict, np.ndarray]:
    """
    Build ordered corners and heading once the two front corners are known.
    """
    if len(front_indices) != 2:
        raise ValueError("front_indices must contain exactly two indices.")

    index_set = set(front_indices)
    if len(index_set) != 2:
        raise ValueError("front_indices must reference two distinct corners.")

    back_indices = [idx for idx in range(4) if idx not in index_set]
    front_a = raw_corners[front_indices[0]]
    front_b = raw_corners[front_indices[1]]
    back_a = raw_corners[back_indices[0]]
    back_b = raw_corners[back_indices[1]]

    front_mid = 0.5 * (front_a + front_b)
    forward = front_mid - center
    forward_mag = float(np.linalg.norm(forward))
    if forward_mag < 1e-6:
        raise ValueError("Front edge midpoint overlaps the car center; heading is ambiguous.")
    forward = (forward / forward_mag).astype(np.float32)

    # In image coordinates (x right, y down), the car's right side is a
    # clockwise 90-degree rotation from the forward direction.
    right = np.array([-forward[1], forward[0]], dtype=np.float32)

    def split_lr(points_pair: tuple[np.ndarray, np.ndarray]):
        p0, p1 = points_pair
        r0 = float((p0 - center) @ right)
        r1 = float((p1 - center) @ right)
        if r0 >= r1:
            return p0, p1
        return p1, p0

    front_right_pt, front_left_pt = split_lr((front_a, front_b))
    back_right_pt, back_left_pt = split_lr((back_a, back_b))

    corners_dict = {
        "front_right": (float(front_right_pt[0]), float(front_right_pt[1])),
        "front_left": (float(front_left_pt[0]), float(front_left_pt[1])),
        "back_right": (float(back_right_pt[0]), float(back_right_pt[1])),
        "back_left": (float(back_left_pt[0]), float(back_left_pt[1])),
    }
    return corners_dict, forward

def _order_corners_and_heading_from_tape_center(
    raw_corners: np.ndarray,
    center: np.ndarray,
    tape_center: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """
    Infer the front from the two corners closest to the detected tape center.

    This matches the intended physical setup: place the tape on the front side,
    then use the tape center to decide which two corners belong to that front.
    """
    corners = raw_corners.astype(np.float32).copy()
    distances = [
        float(np.linalg.norm(corner - tape_center))
        for corner in corners
    ]
    front_indices = sorted(range(4), key=lambda idx: distances[idx])[:2]
    return _corners_and_heading_from_front_pair(corners, center, front_indices)


def detect_physical_car(frame_bgr: np.ndarray) -> dict:
    """
    Detect the physical square car in a stationary overhead webcam image.
    """
    image_height, image_width = frame_bgr.shape[:2]
    body = detect_red_body(frame_bgr)
    front_marker = detect_white_marker_near_body(
        frame_bgr,
        body_bbox=body["bbox"],
        body_center=body["center"],
    )

    marker_center = np.array(front_marker["center"], dtype=np.float32)
    car_corners, forward = _order_corners_and_heading_from_tape_center(
        body["raw_corners"],
        body["center"],
        marker_center,
    )
    heading_deg = _heading_from_vector(forward)
    front_midpoint = (
        0.5 * (np.array(car_corners["front_right"], dtype=np.float32) + np.array(car_corners["front_left"], dtype=np.float32))
    )

    state_array = np.array(
        [
            car_corners["front_right"][0],
            car_corners["front_right"][1],
            car_corners["front_left"][0],
            car_corners["front_left"][1],
            car_corners["back_right"][0],
            car_corners["back_right"][1],
            car_corners["back_left"][0],
            car_corners["back_left"][1],
            heading_deg,
        ],
        dtype=np.float32,
    )

    result = {
        "map_bounds": {
            "left": 0.0,
            "top": 0.0,
            "right": float(image_width - 1),
            "bottom": float(image_height - 1),
        },
        "image_size": {
            "width": int(image_width),
            "height": int(image_height),
        },
        "car_center": (float(body["center"][0]), float(body["center"][1])),
        "car_corners": car_corners,
        "car_body_bbox": {
            "left": body["bbox"][0],
            "top": body["bbox"][1],
            "right": body["bbox"][2],
            "bottom": body["bbox"][3],
        },
        "car_direction": {
            "angle_degrees": heading_deg,
            "forward_vector": (float(forward[0]), float(forward[1])),
            "front_midpoint": (float(front_midpoint[0]), float(front_midpoint[1])),
        },
        "front_marker": {
            "type": front_marker["type"],
            "center": (float(front_marker["center"][0]), float(front_marker["center"][1])),
        },
        "state_array": state_array,
    }

    if "bbox" in front_marker:
        result["front_marker"]["bbox"] = tuple(int(v) for v in front_marker["bbox"])

    return result


def annotate_detection(frame_bgr: np.ndarray, detection: dict) -> np.ndarray:
    """
    Draw the detected robot footprint and heading on top of a frame.
    """
    output = frame_bgr.copy()

    corners = detection["car_corners"]
    polygon = np.array(
        [
            corners["front_right"],
            corners["front_left"],
            corners["back_left"],
            corners["back_right"],
        ],
        dtype=np.int32,
    )
    cv2.polylines(output, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    center = tuple(map(int, detection["car_center"]))
    marker_center = tuple(map(int, detection["front_marker"]["center"]))
    heading_tip = tuple(map(int, detection["car_direction"]["front_midpoint"]))
    cv2.circle(output, center, 4, (0, 255, 255), -1)
    cv2.circle(output, marker_center, 4, (255, 255, 0), -1)
    cv2.circle(output, heading_tip, 4, (0, 128, 255), -1)
    cv2.arrowedLine(output, center, heading_tip, (255, 0, 0), 2, tipLength=0.2)

    if detection["front_marker"]["type"] == "white_tape" and "bbox" in detection["front_marker"]:
        left, top, right, bottom = detection["front_marker"]["bbox"]
        cv2.rectangle(output, (left, top), (right, bottom), (255, 255, 255), 2)

    label_map = {
        "front_right": "FR",
        "front_left": "FL",
        "back_right": "BR",
        "back_left": "BL",
    }
    for name, point in corners.items():
        px = tuple(map(int, point))
        cv2.circle(output, px, 4, (0, 200, 0), -1)
        text_x = px[0] + 6
        text_y = max(14, px[1] - 4)
        _draw_text_box(
            output,
            label_map[name],
            (text_x, text_y),
            font_scale=0.38,
            thickness=1,
            padding=3,
        )

    _draw_text_box(
        output,
        f"heading={detection['car_direction']['angle_degrees']:.1f} deg",
        (12, 24),
        font_scale=0.7,
        thickness=2,
        padding=6,
    )

    _draw_text_box(
        output,
        f"center=({int(detection['car_center'][0])}, {int(detection['car_center'][1])})",
        (12, 52),
        font_scale=0.55,
        thickness=1,
        padding=5,
    )

    return output


def _capture_loop(
    camera_index: int,
    interval: float,
    frame_width: int,
    frame_height: int,
    display: bool,
):
    capture = _open_camera(camera_index)
    _configure_capture(capture, frame_width, frame_height)
    if not capture.isOpened():
        raise _camera_open_error(camera_index)

    try:
        last_time = 0.0
        window_ready = False
        while True:
            now = time.perf_counter()
            if now - last_time < interval:
                if display:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the webcam.")

            detection = detect_physical_car(
                frame,
            )
            print(json.dumps(_to_serializable(detection)))

            if display:
                annotated = annotate_detection(frame, detection)
                if not window_ready:
                    _prepare_preview_window(annotated)
                    window_ready = True
                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            last_time = now
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Detect a square red robot from an overhead webcam."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index for OpenCV VideoCapture.")
    parser.add_argument("--interval", type=float, default=1.0 / 15.0, help="Seconds between processed frames.")
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH, help="Requested camera frame width.")
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_HEIGHT, help="Requested camera frame height.")
    parser.add_argument("--input", default=None, help="Optional image path instead of live webcam input.")
    parser.add_argument(
        "--save-annotated",
        default=None,
        help="Optional output path for the annotated image when using --input.",
    )
    parser.add_argument("--display", action="store_true", help="Show an annotated live preview window.")
    args = parser.parse_args()

    if args.input:
        frame = cv2.imread(args.input)
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.input}")
        detection = detect_physical_car(frame)
        annotated = annotate_detection(frame, detection)
        if args.save_annotated:
            ok = cv2.imwrite(args.save_annotated, annotated)
            if not ok:
                raise RuntimeError(f"Could not write annotated image: {args.save_annotated}")
        if args.display:
            _prepare_preview_window(annotated)
            cv2.imshow(WINDOW_NAME, annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(json.dumps(_to_serializable(detection), indent=2))
        return

    _capture_loop(
        camera_index=args.camera_index,
        interval=args.interval,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        display=args.display,
    )


if __name__ == "__main__":
    main()

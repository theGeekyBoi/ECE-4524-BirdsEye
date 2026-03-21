"""
Screenshot-based detector for the top-down driving window.

The detector locates the game field inside a larger screenshot, then finds:
- car corners (relative to the screenshot)
- car heading in degrees, where 0 degrees points east
- target center (relative to the screenshot)

It uses the simulator's stable colors rather than any direct game-state hooks,
which makes it suitable for a later "screen-in, action-out" control loop.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque

import numpy as np
import pygame

from TestRig import (
    BORDER_COLOR,
    CAR_COLOR,
    CAR_FRONT_COLOR,
    FIELD_COLOR,
    TARGET_COLOR,
)


def _surface_to_image(surface: pygame.Surface) -> np.ndarray:
    """Convert a pygame surface into an (H, W, 3) uint8 NumPy image."""
    rgb = pygame.surfarray.array3d(surface)
    return np.transpose(rgb, (1, 0, 2)).copy()


def _load_image(image_source) -> np.ndarray:
    """Accept a path, pygame surface, or already-loaded RGB NumPy image."""
    if isinstance(image_source, np.ndarray):
        if image_source.ndim != 3 or image_source.shape[2] != 3:
            raise ValueError("NumPy image must have shape (H, W, 3).")
        return image_source.astype(np.uint8, copy=False)

    if isinstance(image_source, pygame.Surface):
        return _surface_to_image(image_source)

    if isinstance(image_source, str):
        surface = pygame.image.load(image_source)
        return _surface_to_image(surface)

    raise TypeError("image_source must be a file path, pygame.Surface, or RGB NumPy array.")


def _color_mask(image: np.ndarray, color, tolerance: int) -> np.ndarray:
    diff = np.abs(image.astype(np.int16) - np.array(color, dtype=np.int16))
    return np.all(diff <= tolerance, axis=2)


def _largest_component(mask: np.ndarray):
    return _component_by_selector(mask, selector="largest")


def _component_by_selector(mask: np.ndarray, selector="largest"):
    """Return a connected component from a boolean mask."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_coords = None
    best_area = -1
    best_score = float("inf")

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            queue = deque([(y, x)])
            visited[y, x] = True
            coords = []

            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))

                if cy > 0 and mask[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = True
                    queue.append((cy - 1, cx))
                if cy + 1 < h and mask[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = True
                    queue.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = True
                    queue.append((cy, cx - 1))
                if cx + 1 < w and mask[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = True
                    queue.append((cy, cx + 1))

            area = len(coords)

            if selector == "largest":
                if area > best_area:
                    best_area = area
                    best_coords = coords
            else:
                centroid = np.mean(coords, axis=0)
                score = selector(centroid, area)
                if score < best_score:
                    best_score = score
                    best_coords = coords

    if best_coords is None:
        return None

    component_mask = np.zeros_like(mask, dtype=bool)
    ys = []
    xs = []
    for y, x in best_coords:
        component_mask[y, x] = True
        ys.append(y)
        xs.append(x)

    return {
        "mask": component_mask,
        "area": len(best_coords),
        "bbox": (min(xs), min(ys), max(xs), max(ys)),
        "centroid_yx": (float(np.mean(ys)), float(np.mean(xs))),
    }


def locate_playfield(image: np.ndarray, tolerance: int = 8):
    """
    Detect the dark driving field inside a full screenshot.

    Returns an inclusive bounding box: (left, top, right, bottom).
    """
    field_mask = _color_mask(image, FIELD_COLOR, tolerance)
    coords = np.argwhere(field_mask)
    if coords.size == 0:
        raise ValueError("Could not locate the playfield color in the screenshot.")

    ys = coords[:, 0]
    xs = coords[:, 1]
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _extract_region(image: np.ndarray, bbox):
    left, top, right, bottom = bbox
    return image[top:bottom + 1, left:right + 1]


def detect_target(image: np.ndarray, playfield_bbox):
    """Detect the cyan target and return its center relative to the screenshot."""
    region = _extract_region(image, playfield_bbox)
    target_mask = _color_mask(region, TARGET_COLOR, tolerance=36)
    component = _largest_component(target_mask)
    if component is None:
        raise ValueError("Could not detect the target in the screenshot.")

    cy, cx = component["centroid_yx"]
    left, top, _, _ = playfield_bbox
    radius = math.sqrt(component["area"] / math.pi)
    return {
        "coordinates": (left + cx, top + cy),
        "radius": radius,
        "bbox": (
            left + component["bbox"][0],
            top + component["bbox"][1],
            left + component["bbox"][2],
            top + component["bbox"][3],
        ),
        "pixel_area": component["area"],
    }


def detect_car(image: np.ndarray, playfield_bbox):
    """Detect the yellow car body, blue heading marker, and inferred corners."""
    region = _extract_region(image, playfield_bbox)
    left, top, _, _ = playfield_bbox

    body_mask = _color_mask(region, CAR_COLOR, tolerance=36)
    body_component = _largest_component(body_mask)
    if body_component is None:
        raise ValueError("Could not detect the car body in the screenshot.")

    body_ys, body_xs = np.nonzero(body_component["mask"])
    center_x = float(body_xs.mean())
    center_y = float(body_ys.mean())

    front_mask = _color_mask(region, CAR_FRONT_COLOR, tolerance=42)

    def front_selector(centroid_yx, area):
        dy = centroid_yx[0] - center_y
        dx = centroid_yx[1] - center_x
        # Favor front-marker-sized components near the car body.
        return math.hypot(dx, dy) + abs(area - 40) * 0.25

    front_component = _component_by_selector(front_mask, selector=front_selector)
    if front_component is None:
        raise ValueError("Could not detect the car heading marker in the screenshot.")

    front_y, front_x = front_component["centroid_yx"]
    vec_x = front_x - center_x
    vec_y = front_y - center_y
    mag = math.hypot(vec_x, vec_y)
    if mag < 1e-6:
        raise ValueError("Car heading marker was detected, but direction was degenerate.")

    forward = np.array([vec_x / mag, vec_y / mag], dtype=np.float64)
    right = np.array([forward[1], -forward[0]], dtype=np.float64)

    points = np.column_stack((body_xs.astype(np.float64) - center_x, body_ys.astype(np.float64) - center_y))
    forward_proj = points @ forward
    right_proj = points @ right

    f_max = float(forward_proj.max())
    f_min = float(forward_proj.min())
    r_max = float(right_proj.max())
    r_min = float(right_proj.min())

    center = np.array([center_x, center_y], dtype=np.float64)

    def corner(f_amt, r_amt):
        pt = center + forward * f_amt + right * r_amt
        return (left + float(pt[0]), top + float(pt[1]))

    heading_deg = (math.degrees(math.atan2(-forward[1], forward[0])) + 360.0) % 360.0

    return {
        "center": (left + center_x, top + center_y),
        "angle_degrees": heading_deg,
        "forward_vector": (float(forward[0]), float(forward[1])),
        "corners": {
            "front_right": corner(f_max, r_max),
            "front_left": corner(f_max, r_min),
            "back_right": corner(f_min, r_max),
            "back_left": corner(f_min, r_min),
        },
        "body_bbox": (
            left + body_component["bbox"][0],
            top + body_component["bbox"][1],
            left + body_component["bbox"][2],
            top + body_component["bbox"][3],
        ),
    }


def build_detection_result(image_source):
    """
    Main entry point.

    Returns both a structured dictionary and a compact flat state array:
    [fr_x, fr_y, fl_x, fl_y, br_x, br_y, bl_x, bl_y, heading_deg, target_x, target_y]
    """
    image = _load_image(image_source)
    playfield_bbox = locate_playfield(image)
    car = detect_car(image, playfield_bbox)
    target = detect_target(image, playfield_bbox)

    corners = car["corners"]
    state_array = np.array(
        [
            corners["front_right"][0],
            corners["front_right"][1],
            corners["front_left"][0],
            corners["front_left"][1],
            corners["back_right"][0],
            corners["back_right"][1],
            corners["back_left"][0],
            corners["back_left"][1],
            car["angle_degrees"],
            target["coordinates"][0],
            target["coordinates"][1],
        ],
        dtype=np.float32,
    )

    return {
        "playfield_bbox": playfield_bbox,
        "car_corners": corners,
        "car_direction": {
            "angle_degrees": car["angle_degrees"],
            "forward_vector": car["forward_vector"],
        },
        "target": {
            "coordinates": target["coordinates"],
            "radius": target["radius"],
        },
        "state_array": state_array,
    }


def detection_to_model_state(result: dict) -> np.ndarray:
    """
    Convert screenshot-relative detections into the 22-feature state used by CarEnv.
    """
    left, top, right, bottom = result["playfield_bbox"]
    field_width = float(right - left + 1)
    field_height = float(bottom - top + 1)
    field_diag = math.hypot(field_width, field_height)

    corners = result["car_corners"]
    direction = result["car_direction"]
    target = result["target"]

    def rel(point):
        return (point[0] - left, point[1] - top)

    fr = rel(corners["front_right"])
    fl = rel(corners["front_left"])
    br = rel(corners["back_right"])
    bl = rel(corners["back_left"])

    car_center_x = (fr[0] + fl[0] + br[0] + bl[0]) / 4.0
    car_center_y = (fr[1] + fl[1] + br[1] + bl[1]) / 4.0

    tx, ty = rel(target["coordinates"])
    dx = tx - car_center_x
    dy = ty - car_center_y
    dist = math.hypot(dx, dy)

    angle_rad = math.radians(direction["angle_degrees"])
    wall_left = car_center_x / field_width
    wall_right = (field_width - car_center_x) / field_width
    wall_top = car_center_y / field_height
    wall_bottom = (field_height - car_center_y) / field_height

    return np.array(
        [
            fr[0] / field_width,
            fr[1] / field_height,
            fl[0] / field_width,
            fl[1] / field_height,
            br[0] / field_width,
            br[1] / field_height,
            bl[0] / field_width,
            bl[1] / field_height,
            math.sin(angle_rad),
            math.cos(angle_rad),
            direction["forward_vector"][0],
            direction["forward_vector"][1],
            tx / field_width,
            ty / field_height,
            target["radius"] / max(field_width, field_height),
            dist / field_diag,
            dx / field_diag,
            dy / field_diag,
            wall_left,
            wall_right,
            wall_top,
            wall_bottom,
        ],
        dtype=np.float32,
    )


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    return obj


def main():
    parser = argparse.ArgumentParser(description="Detect car and target from a screenshot.")
    parser.add_argument("--input", required=True, help="Path to a screenshot image.")
    args = parser.parse_args()

    pygame.init()
    result = build_detection_result(args.input)
    print(json.dumps(_to_serializable(result), indent=2))


if __name__ == "__main__":
    main()

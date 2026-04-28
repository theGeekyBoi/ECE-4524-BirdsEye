"""
Physical car detector for a stationary overhead webcam.

Assumptions:
- the full camera frame is the drivable map
- the robot has a bright red top cover
- the front is marked by a small white tape marker
- the car footprint is approximately square in the image

Outputs are pixel coordinates relative to the full camera image.

Fixes vs original:
- Red mask now uses BOTH hue ranges (0-10 and 160-180) and requires higher
  saturation to exclude wheels/shadows that bled into the bbox.
- Heading is computed directly as (tape_center - car_center), normalised.
  No corner-pair ambiguity, no 180-degree flip.
- Corners are labeled by projecting onto the (forward, right) axes of that
  direct heading, so FR/FL/BR/BL are always consistent with the arrow.
- Last-good-detection fallback prevents crashes on occluded frames.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import time

import cv2
import numpy as np

WINDOW_NAME = "Physical Car Detection"
DEFAULT_FRAME_WIDTH  = 1920
DEFAULT_FRAME_HEIGHT = 1080

# ── Smoothing parameters ─────────────────────────────────────────────────────
HEADING_ALPHA = 0.5
POSITION_ALPHA = 0.5
MIN_HEADING_CHANGE_DEG = 4.0
# ─────────────────────────────────────────────────────────────────────────────


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def _draw_text_box(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float = 0.7,
    text_color: tuple[int, int, int] = (255, 255, 255),
    box_color:  tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    padding: int = 6,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(
        image,
        (max(0, x - padding),          max(0, y - th - padding)),
        (min(image.shape[1]-1, x+tw+padding), min(image.shape[0]-1, y+baseline+padding)),
        box_color, -1,
    )
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def _heading_from_vector(fwd: np.ndarray) -> float:
    """Unit forward vector (x-right, y-down) -> compass angle [0, 360)."""
    return (math.degrees(math.atan2(-fwd[1], fwd[0])) + 360.0) % 360.0


def _angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(v1 @ v2, -1.0, 1.0))))


def _running_in_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower()


def _camera_open_error(idx: int) -> RuntimeError:
    if os.name == "posix" and _running_in_wsl() and not any(
        os.path.exists(f"/dev/video{i}") for i in range(8)
    ):
        return RuntimeError(
            f"Could not open camera {idx}. No /dev/video* devices in WSL2. "
            "Run from Windows Python or attach via usbipd."
        )
    return RuntimeError(f"Could not open camera index {idx}.")


def _open_camera(idx: int) -> cv2.VideoCapture:
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    return cv2.VideoCapture(idx)


def _configure_capture(cap: cv2.VideoCapture, w: int, h: int):
    if hasattr(cv2, "VideoWriter_fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


def _prepare_preview_window(image: np.ndarray):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, image.shape[1], image.shape[0])


# ── Colour masks ──────────────────────────────────────────────────────────────

def _build_red_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Isolate the red robot body.

    Uses BOTH hue wrap-around ranges (0-10 deg and 160-180 deg) with
    a reasonably high saturation floor so that pale wheels, shadows, and
    grey carpet are excluded.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv,
                        np.array([0,   80,  80], np.uint8),
                        np.array([10, 255, 255], np.uint8))
    mask2 = cv2.inRange(hsv,
                        np.array([160, 80,  80], np.uint8),
                        np.array([180, 255, 255], np.uint8))
    mask = cv2.bitwise_or(mask1, mask2)

    k5   = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    return mask


def _build_white_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([0,   0, 170], np.uint8),
                       np.array([180, 90, 255], np.uint8))
    k3   = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)
    return mask



# ── Detection primitives ──────────────────────────────────────────────────────

def detect_red_body(frame_bgr: np.ndarray, min_area: float = 400.0) -> dict:
    """
    Find the largest red region and fit a rotated rectangle to it.

    Also returns a ``hull_mask`` — a filled convex hull of the red contour
    drawn at full-frame resolution.  This mask is used downstream to restrict
    the white-tape search to pixels that lie strictly inside the car footprint,
    eliminating false detections from bright background regions that happen to
    fall inside the loose axis-aligned bounding box.
    """
    mask        = _build_red_mask(frame_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No red regions found in frame.")

    contour = max(contours, key=cv2.contourArea)
    area    = float(cv2.contourArea(contour))
    if area < min_area:
        raise ValueError(f"Red region too small (area={area:.1f}).")

    rect        = cv2.minAreaRect(contour)
    raw_corners = cv2.boxPoints(rect).astype(np.float32)
    center      = np.array(rect[0], dtype=np.float32)
    x, y, w, h  = cv2.boundingRect(contour)

    # Build a filled convex-hull mask at full-frame resolution.
    # Using the rotated-rect corners (boxPoints) gives a clean quadrilateral
    # that hugs the car without stray noise pixels from the raw contour.
    hull_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    hull_pts  = raw_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillConvexPoly(hull_mask, hull_pts, 255)

    return {
        "mask":        mask,
        "hull_mask":   hull_mask,
        "contour":     contour,
        "area":        area,
        "center":      center,
        "raw_corners": raw_corners,
        "rect":        rect,
        "bbox":        (int(x), int(y), int(x+w-1), int(y+h-1)),
    }


def detect_white_marker_near_body(frame_bgr, body_center, hull_mask, red_mask, min_area=12.0) -> dict:
    """
    Find the white tape marker strictly inside the car's rotated footprint.

    Instead of cropping an axis-aligned bounding box (which includes large
    background triangles when the car is rotated), we:

      1. Build the white mask at full-frame resolution.
      2. AND it with ``hull_mask`` — the filled rotated-rect quadrilateral of
         the red body.  Only pixels inside the actual car silhouette survive.
      3. Pick the best surviving contour by distance-from-center + area score.

    This eliminates false detections from bright carpet / walls that happen to
    fall inside the old loose bbox but outside the real car footprint.
    """
    white_mask_full = _build_white_mask(frame_bgr)
    # shrink RED mask (not recompute)
    k = np.ones((5, 5), np.uint8)
    inner_red = cv2.erode(red_mask, k)

    # combine with hull
    safe_roi = cv2.bitwise_and(inner_red, hull_mask)

    # detect white inside safe ROI
    masked = cv2.bitwise_and(white_mask_full, safe_roi)

    # FALLBACK if erosion killed everything
    if cv2.countNonZero(masked) < 10:
        masked = cv2.bitwise_and(white_mask_full, hull_mask)
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No white marker found inside car footprint.")

    best, best_score = None, -float("inf")

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        m = cv2.moments(cnt)
        if abs(m["m00"]) < 1e-6:
            continue
        cx  = float(m["m10"] / m["m00"])
        cy  = float(m["m01"] / m["m00"])
        ctr = np.array([cx, cy], np.float32)
        score = 2.0 * area - 0.02 * np.linalg.norm(ctr - body_center)
        if score > best_score:
            best_score = score
            xb, yb, wb, hb = cv2.boundingRect(cnt)
            best = {
                "type":   "white_tape",
                "center": ctr,
                "area":   area,
                "bbox":   (int(xb), int(yb), int(xb+wb-1), int(yb+hb-1)),
            }

    if best is None:
        raise ValueError("No usable white marker found inside car footprint.")
    return best


def _label_corners(raw_corners: np.ndarray, center: np.ndarray, forward: np.ndarray) -> dict:
    """
    Assign FR/FL/BR/BL to the four rotated-rect corners by projecting them
    onto the (forward, right) axes derived from the tape-based heading.

    This replaces the old 'two closest corners to tape' heuristic which
    caused 180-degree flips when the tape was near an edge midpoint.
    """
    # right = CW 90-degree rotation of forward (image coords, y-down)
    right = np.array([-forward[1], forward[0]], np.float32)

    labeled: dict[str, tuple[float, float]] = {}
    for corner in raw_corners:
        rel        = corner - center
        f_proj     = float(rel @ forward)
        r_proj     = float(rel @ right)
        fb         = "front" if f_proj >= 0 else "back"
        lr         = "right" if r_proj >= 0 else "left"
        key        = f"{fb}_{lr}"
        mag2       = float(f_proj**2 + r_proj**2)
        if key not in labeled or mag2 > labeled[key][2]:
            labeled[key] = (float(corner[0]), float(corner[1]), mag2)

    # Strip the internal magnitude field and guarantee all four keys exist.
    result: dict[str, tuple[float, float]] = {}
    for k in ("front_right", "front_left", "back_right", "back_left"):
        if k in labeled:
            result[k] = (labeled[k][0], labeled[k][1])
        else:
            result[k] = (float(center[0]), float(center[1]))
    return result


# ── Main detection ────────────────────────────────────────────────────────────

def detect_physical_car(frame_bgr: np.ndarray) -> dict:
    """
    Detect the physical square car and return its pose.

    Key design decision:
      forward = normalise(tape_center - car_center)

    This is unambiguous regardless of car orientation and eliminates the
    180-degree flip that plagued the old corner-pair selection approach.
    """
    fh, fw = frame_bgr.shape[:2]

    body         = detect_red_body(frame_bgr)
    front_marker = detect_white_marker_near_body(
        frame_bgr,
        body_center=body["center"],
        hull_mask=body["hull_mask"],
        red_mask=body["mask"],
    )

    tape_center = np.array(front_marker["center"], np.float32)
    car_center  = body["center"]
    raw_fwd     = tape_center - car_center
    mag         = float(np.linalg.norm(raw_fwd))
    if mag < 1.0:
        raise ValueError("Tape too close to car center; heading ambiguous.")
    forward = (raw_fwd / mag).astype(np.float32)

    car_corners   = _label_corners(body["raw_corners"], car_center, forward)
    heading_deg   = _heading_from_vector(forward)
    front_mid     = 0.5 * (
        np.array(car_corners["front_right"], np.float32)
        + np.array(car_corners["front_left"],  np.float32)
    )

    state_array = np.array([
        car_corners["front_right"][0], car_corners["front_right"][1],
        car_corners["front_left"][0],  car_corners["front_left"][1],
        car_corners["back_right"][0],  car_corners["back_right"][1],
        car_corners["back_left"][0],   car_corners["back_left"][1],
        heading_deg,
    ], dtype=np.float32)

    return {
        "map_bounds":  {"left": 0.0, "top": 0.0,
                        "right": float(fw-1), "bottom": float(fh-1)},
        "image_size":  {"width": int(fw), "height": int(fh)},
        "car_center":  (float(car_center[0]), float(car_center[1])),
        "car_corners": car_corners,
        "car_body_bbox": {
            "left":   body["bbox"][0], "top":    body["bbox"][1],
            "right":  body["bbox"][2], "bottom": body["bbox"][3],
        },
        "car_direction": {
            "angle_degrees":  heading_deg,
            "forward_vector": (float(forward[0]), float(forward[1])),
            "front_midpoint": (float(front_mid[0]), float(front_mid[1])),
        },
        "front_marker": {
            "type":   front_marker["type"],
            "center": (float(tape_center[0]), float(tape_center[1])),
            "bbox":   tuple(int(v) for v in front_marker["bbox"]),
        },
        "state_array": state_array,
    }, body  # body is returned so callers can pass hull_mask to annotate_detection


# ── Smoothing ─────────────────────────────────────────────────────────────────

class DetectionSmoother:
    """
    EMA smoothing on heading (vector space) and position.

    Heading is smoothed as a unit vector so there is no 0/360 wraparound
    discontinuity.  A deadband discards sub-threshold noise entirely.
    """

    def __init__(
        self,
        heading_alpha:          float = HEADING_ALPHA,
        position_alpha:         float = POSITION_ALPHA,
        min_heading_change_deg: float = MIN_HEADING_CHANGE_DEG,
    ):
        self.ha      = heading_alpha
        self.pa      = position_alpha
        self.min_deg = min_heading_change_deg
        self._fwd:    np.ndarray | None = None
        self._center: np.ndarray | None = None
        self._last_out: None
        self.max_center_jump_px = 150.0

    def reset(self):
        self._fwd = self._center = None

    def update(self, det: dict) -> dict:
        raw_fwd = np.array(det["car_direction"]["forward_vector"], np.float32)
        raw_center = np.array(det["car_center"], np.float32)
        if self._center is not None:
            jump = float(np.linalg.norm(raw_center - self._center))
            if jump > self.max_center_jump_px and self._last_out is not None:
                return self._last_out
        raw_corners = np.array(list(det["car_corners"].values()), np.float32)
        dists = np.linalg.norm(raw_corners - raw_center, axis=1)
        half_diag = float(np.mean(dists))

        if self._center is None:
            self._center = raw_center.copy()
        else:
            self._center = self.pa * raw_center + (1.0 - self.pa) * self._center

        if self._fwd is None:
            self._fwd = raw_fwd.copy()
        else:
            change = _angle_between_vectors_deg(self._fwd, raw_fwd)
            if change >= self.min_deg:
                blended = self.ha * raw_fwd + (1.0 - self.ha) * self._fwd
                n = float(np.linalg.norm(blended))
                if n > 1e-6:
                    self._fwd = (blended / n).astype(np.float32)

        fwd = self._fwd
        right = np.array([-fwd[1], fwd[0]], np.float32)

        # Rebuild a stable square from smoothed center + smoothed heading.
        half_side = half_diag / math.sqrt(2.0)

        fc = self._center + fwd * half_side
        bc = self._center - fwd * half_side

        fr = fc + right * half_side
        fl = fc - right * half_side
        br = bc + right * half_side
        bl = bc - right * half_side

        sc = {
            "front_right": (float(fr[0]), float(fr[1])),
            "front_left":  (float(fl[0]), float(fl[1])),
            "back_right":  (float(br[0]), float(br[1])),
            "back_left":   (float(bl[0]), float(bl[1])),
        }

        fmp = self._center + fwd * half_side
        smooth_heading = _heading_from_vector(fwd)

        out = copy.deepcopy(det)
        out["car_center"] = (float(self._center[0]), float(self._center[1]))
        out["car_corners"] = sc
        out["car_direction"]["angle_degrees"] = smooth_heading
        out["car_direction"]["forward_vector"] = (float(fwd[0]), float(fwd[1]))
        out["car_direction"]["front_midpoint"] = (float(fmp[0]), float(fmp[1]))

        out["state_array"] = np.array([
            sc["front_right"][0], sc["front_right"][1],
            sc["front_left"][0],  sc["front_left"][1],
            sc["back_right"][0],  sc["back_right"][1],
            sc["back_left"][0],   sc["back_left"][1],
            smooth_heading,
        ], dtype=np.float32)

        self._last_out = out
        return out


# ── Annotation ────────────────────────────────────────────────────────────────

def annotate_detection(frame_bgr: np.ndarray, det: dict, body: dict | None = None) -> np.ndarray:
    out = frame_bgr.copy()

    # Optional: draw the hull mask outline in purple so you can verify it hugs the car
    if body is not None and "hull_mask" in body:
        hull_contours, _ = cv2.findContours(body["hull_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, hull_contours, -1, (180, 0, 180), 1)

    c    = det["car_corners"]

    poly = np.array([c["front_right"], c["front_left"],
                     c["back_left"],   c["back_right"]], dtype=np.int32)
    cv2.polylines(out, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

    ctr  = tuple(map(int, det["car_center"]))
    tip  = tuple(map(int, det["car_direction"]["front_midpoint"]))
    tape = tuple(map(int, det["front_marker"]["center"]))

    cv2.circle(out, ctr,  4, (0, 255, 255), -1)  # cyan   = body center
    cv2.circle(out, tape, 4, (255, 255, 0), -1)  # yellow = tape centroid
    cv2.circle(out, tip,  4, (0, 128, 255), -1)  # orange = front-edge midpoint
    cv2.arrowedLine(out, ctr, tip, (255, 0, 0), 2, tipLength=0.25)

    if "bbox" in det["front_marker"]:
        l, t, r, b = det["front_marker"]["bbox"]
        cv2.rectangle(out, (l, t), (r, b), (255, 255, 255), 2)

    labels = {"front_right": "FR", "front_left": "FL",
              "back_right":  "BR", "back_left":  "BL"}
    for name, pt in c.items():
        px = tuple(map(int, pt))
        cv2.circle(out, px, 4, (0, 200, 0), -1)
        _draw_text_box(out, labels[name], (px[0]+6, max(14, px[1]-4)),
                       font_scale=0.38, thickness=1, padding=3)

    _draw_text_box(out, f"heading={det['car_direction']['angle_degrees']:.1f} deg",
                   (12, 24), font_scale=0.7, thickness=2)
    _draw_text_box(out, f"center=({int(det['car_center'][0])}, {int(det['car_center'][1])})",
                   (12, 52), font_scale=0.55, thickness=1)
    return out


# ── Capture loop ──────────────────────────────────────────────────────────────

def _capture_loop(
    camera_index: int, interval: float,
    frame_width: int, frame_height: int, display: bool,
    heading_alpha: float, position_alpha: float,
    min_heading_change_deg: float,
):
    cap = _open_camera(camera_index)
    _configure_capture(cap, frame_width, frame_height)
    if not cap.isOpened():
        raise _camera_open_error(camera_index)

    smoother  = DetectionSmoother(heading_alpha, position_alpha, min_heading_change_deg)
    last_good = None

    try:
        last_t, window_ready = 0.0, False
        while True:
            now = time.perf_counter()
            if now - last_t < interval:
                if display and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from webcam.")

            try:
                raw, body = detect_physical_car(frame)
                smoothed  = smoother.update(raw)
                last_good = (smoothed, body)
            except ValueError:
                if last_good is not None:
                    smoothed, body = last_good
                else:
                    last_t = now
                    continue

            print(json.dumps(_to_serializable(smoothed)))

            if display:
                annotated = annotate_detection(frame, smoothed, body)
                if not window_ready:
                    _prepare_preview_window(annotated)
                    window_ready = True
                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            last_t = now
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Detect a square red robot from an overhead webcam.")
    p.add_argument("--camera-index",       type=int,   default=1)
    p.add_argument("--interval",           type=float, default=0.1)
    p.add_argument("--frame-width",        type=int,   default=DEFAULT_FRAME_WIDTH)
    p.add_argument("--frame-height",       type=int,   default=DEFAULT_FRAME_HEIGHT)
    p.add_argument("--input",              default=None,
                   help="Image path to process instead of live webcam.")
    p.add_argument("--save-annotated",     default=None)
    p.add_argument("--display",            action="store_true")
    p.add_argument("--heading-alpha",      type=float, default=HEADING_ALPHA,
                   help="EMA weight for new heading [0=frozen .. 1=raw].")
    p.add_argument("--position-alpha",     type=float, default=POSITION_ALPHA,
                   help="EMA weight for new position [0=frozen .. 1=raw].")
    p.add_argument("--min-heading-change", type=float, default=MIN_HEADING_CHANGE_DEG,
                   help="Heading deadband in degrees.")
    args = p.parse_args()

    smoother = DetectionSmoother(args.heading_alpha, args.position_alpha, args.min_heading_change)

    if args.input:
        frame = cv2.imread(args.input)
        if frame is None:
            raise RuntimeError(f"Could not read: {args.input}")
        det, body = detect_physical_car(frame)
        smoothed  = smoother.update(det)
        annotated = annotate_detection(frame, smoothed, body)
        if args.save_annotated:
            if not cv2.imwrite(args.save_annotated, annotated):
                raise RuntimeError(f"Could not write: {args.save_annotated}")
        if args.display:
            _prepare_preview_window(annotated)
            cv2.imshow(WINDOW_NAME, annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(json.dumps(_to_serializable(smoothed), indent=2))
        return

    _capture_loop(
        camera_index=args.camera_index, interval=args.interval,
        frame_width=args.frame_width,   frame_height=args.frame_height,
        display=args.display,
        heading_alpha=args.heading_alpha, position_alpha=args.position_alpha,
        min_heading_change_deg=args.min_heading_change,
    )


if __name__ == "__main__":
    main()
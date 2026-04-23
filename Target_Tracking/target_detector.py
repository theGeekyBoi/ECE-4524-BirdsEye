import time

import cv2
import numpy as np


# Default HSV target color and tolerance used for the colored goal marker.
# H range: 0-179, S range: 0-255, V range: 0-255.
DEFAULT_TARGET_HSV = np.array([80, 252, 102], dtype=np.int16)
DEFAULT_TOLERANCE = np.array([18, 80, 60], dtype=np.int16)


def _hsv_bounds(target_hsv: np.ndarray, tolerance: np.ndarray):
    lower = np.clip(target_hsv - tolerance, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
    upper = np.clip(target_hsv + tolerance, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
    return lower, upper


def detect_target(
    frame_bgr: np.ndarray,
    target_hsv: np.ndarray = DEFAULT_TARGET_HSV,
    tolerance: np.ndarray = DEFAULT_TOLERANCE,
) -> dict:
    """
    Detect a colored target blob in a BGR frame using an HSV threshold.

    Returns a dict with:
      - center: (x, y) of the min-enclosing circle around the largest contour
      - radius: float radius of that enclosing circle
      - mask:   binary uint8 mask of pixels matching the HSV range

    Raises ValueError if no matching contour is found.
    """
    lower, upper = _hsv_bounds(
        np.asarray(target_hsv, dtype=np.int16),
        np.asarray(tolerance, dtype=np.int16),
    )
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No target color detected in frame.")

    largest_contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
    return {
        "center": (float(cx), float(cy)),
        "radius": float(radius),
        "mask": mask,
    }


def main() -> None:
    # Camera index 1 is the expected external/USB camera.
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("USB Camera not plugged in.")
        return

    window_name = "Binary HSV Mask"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            try:
                detection = detect_target(frame)
                mask = detection["mask"]
                center = (
                    int(detection["center"][0]),
                    int(detection["center"][1]),
                )
                radius_px = int(detection["radius"])
                if radius_px > 0:
                    cv2.circle(mask, center, radius_px, 255, 2)
                    print(
                        f"center=({center[0]}, {center[1]}), "
                        f"radius={detection['radius']:.2f}"
                    )
            except ValueError:
                # No target detected this frame; show the raw mask instead.
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower, upper = _hsv_bounds(DEFAULT_TARGET_HSV, DEFAULT_TOLERANCE)
                mask = cv2.inRange(hsv, lower, upper)

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0.0:
                fps = 1.0 / dt

            cv2.putText(
                mask,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                255,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or Esc
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

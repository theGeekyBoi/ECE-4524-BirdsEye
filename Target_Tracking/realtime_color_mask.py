import time

import cv2
import numpy as np


def main() -> None:
    # Camera index 1 is the expected external/USB camera.
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("USB Camera not plugged in.")
        return

    # ---------------- USER TUNING SECTION ----------------
    # Store your desired HSV color target here as [H, S, V].
    # H range: 0-179, S range: 0-255, V range: 0-255.
    target_hsv = np.array([60, 180, 180], dtype=np.int16)
    # Tolerance controls how wide the accepted range is around target_hsv.
    # Increase values if detection is too strict (missing pixels).
    # Decrease values if detection is too loose (too many false positives).
    tolerance = np.array([10, 60, 60], dtype=np.int16)
    # -----------------------------------------------------

    # Compute HSV bounds once from target + tolerance.
    # For dynamic runtime retuning, recompute these after changing target/tolerance.
    lower = np.clip(target_hsv - tolerance, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
    upper = np.clip(target_hsv + tolerance, [0, 0, 0], [179, 255, 255]).astype(np.uint8)

    window_name = "Binary HSV Mask"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(cx), int(cy))
                radius_px = int(radius)

                if radius_px > 0:
                    cv2.circle(mask, center, radius_px, 255, 2)
                    print(f"center=({center[0]}, {center[1]}), radius={radius:.2f}")

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

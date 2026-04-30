"""
calibrate_car.py

Standalone calibration helper for the BirdsEye project.

Measures, against the same overhead webcam + 900x600 resized frame the policy
sees at runtime:

    Phase 1 - Footprint
        Average raw rotated-rectangle corners from detect_physical_car over N
        frames; report mean / stdev of length and width in pixels. Compare to
        TestRig.CAR_LENGTH / CAR_WIDTH.

    Phase 2 - Forward speed (action 0)
    Phase 3 - Backward speed (action 1)
    Phase 4 - CCW rotation speed (action 2)
    Phase 5 - CW rotation speed (action 3)

For each motion phase: snapshot car pose, send the action via the existing
Bluetooth serial path, sleep for the phase's motion duration
(--linear-motion-duration for forward/backward, --angular-motion-duration
for CCW/CW; or --motion-duration to set both at once), send action 4 (stop),
let the car settle, snapshot again, derive linear speed (px/s, signed onto
the pre-motion forward axis) or angular speed (deg/s, atan2 of cross/dot on
forward unit vectors so 0/360 wrap-around does not poison the result).

Each phase runs --trials independent trials and reports mean +/- stdev plus
min/max. A small drift side-metric is reported during the rotation phases
because the real RC car will not pivot in place and we will need that number
when we extend the simulator.

Output is plain stdout, with an optional --gui mode that opens an OpenCV
window so you can watch the annotated detection (heading arrow, oriented
polygon, FR/FL/BR/BL labels) live during every phase, and freezes briefly on
each trial's t0/t1 snapshots with the displacement vector overlaid. Press
'q' in the GUI window to quit early.

Usage:
    python calibrate_car.py
    python calibrate_car.py --gui                  # adds visualization window
    python calibrate_car.py --no-serial            # perception path only
    python calibrate_car.py --motion-duration 1.5 --trials 5
    python calibrate_car.py --linear-motion-duration 0.3 --angular-motion-duration 1.2
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "Target_Tracking"))
sys.path.insert(0, os.path.join(HERE, "Bluetooth_Comms"))

from Target_Tracking.car_detector import (  # noqa: E402
    DetectionSmoother,
    _configure_capture,
    _open_camera,
    annotate_detection,
    detect_physical_car,
)
from Target_Tracking.state_generator import _resize_to_map  # noqa: E402
from Bluetooth_Comms.remote_control import COMMANDS, PORT, send_command  # noqa: E402

import TestRig  # noqa: E402  (only used to read sim constants for the report)


GUI_WINDOW_NAME = "calibrate_car (GUI)"
# Smoother warmup counts. The smoother in car_detector.py uses an EMA on
# unit-vector forward + a 4-degree deadband, so it needs several consecutive
# raw updates to converge after a heading change. We warm it up explicitly
# right before each t0/t1 read so the smoothed forward we measure with is
# representative of the current direction, not of a stale earlier one.
SMOOTHER_PRE_MOTION_WARMUP_FRAMES = 12
SMOOTHER_POST_MOTION_WARMUP_FRAMES = 18
# In headless motion mode (no GUI live feed) we still need to feed the smoother
# during the action's hold time, otherwise its center-jump rejection
# (max_center_jump_px = 150) can throw out the post-motion frame entirely.
HEADLESS_MOTION_FEED_INTERVAL_S = 0.05
# Most USB webcams buffer several frames internally. cv2.VideoCapture.read()
# returns the OLDEST queued frame, so without draining we routinely measure
# pre-motion frames at "t1" and silently report zero displacement even though
# the car physically moved. We attempt to set CAP_PROP_BUFFERSIZE=1 at open
# time AND drain this many frames immediately before any measurement read.
CAMERA_BUFFER_DRAIN = 10


SIM_CAR_LENGTH = TestRig.CAR_LENGTH
SIM_CAR_WIDTH = TestRig.CAR_WIDTH
SIM_CAR_SPEED = TestRig.CAR_SPEED
SIM_CAR_ROT_SPEED = TestRig.CAR_ROT_SPEED


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the physical RC car's footprint and per-action linear / "
            "angular speeds in the same 900x600 resized frame the DQN policy "
            "sees at runtime."
        )
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="Webcam index for OpenCV VideoCapture (default: 1).",
    )
    parser.add_argument(
        "--port",
        default=PORT,
        help=f"Serial port for the car link (default: {PORT}).",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=9600,
        help="Serial baud rate (default: 9600).",
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help=(
            "Skip opening the serial port. The motion phases still run their "
            "timing but no command is actually sent to the car. Useful for a "
            "perception-only sanity check."
        ),
    )
    parser.add_argument(
        "--linear-motion-duration",
        type=float,
        default=0.5,
        help=(
            "Seconds to hold the action during each forward/backward trial "
            "(default: 0.5). Shorter than the angular duration so the car "
            "does not run out of room on the table."
        ),
    )
    parser.add_argument(
        "--angular-motion-duration",
        type=float,
        default=1.0,
        help=(
            "Seconds to hold the action during each CCW/CW rotation trial "
            "(default: 1.0). Kept full-length so the swept arc is large "
            "enough for a clean deg/s estimate."
        ),
    )
    parser.add_argument(
        "--motion-duration",
        type=float,
        default=None,
        help=(
            "Backward-compat shortcut: if provided, overrides BOTH "
            "--linear-motion-duration and --angular-motion-duration with "
            "this single value."
        ),
    )
    parser.add_argument(
        "--settle-duration",
        type=float,
        default=0.3,
        help=(
            "Seconds to wait after sending action 4 (stop) before snapshotting "
            "the post-motion pose (default: 0.3)."
        ),
    )
    parser.add_argument(
        "--countdown",
        type=float,
        default=3.0,
        help=(
            "Seconds of countdown printed between phases so you have time to "
            "reposition the car if needed (default: 3.0)."
        ),
    )
    parser.add_argument(
        "--footprint-frames",
        type=int,
        default=30,
        help="Number of frames to average for the footprint phase (default: 30).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per motion phase (default: 3).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help=(
            "Open an OpenCV window that shows the annotated detection live "
            "(heading arrow, oriented polygon, corner labels) during every "
            "phase, and briefly freezes on each trial's t0/t1 snapshots with "
            "the displacement vector drawn between them. Press 'q' to quit."
        ),
    )
    parser.add_argument(
        "--gui-freeze",
        type=float,
        default=0.8,
        help=(
            "Seconds to freeze on each trial's t0/t1 comparison frame in GUI "
            "mode (default: 0.8). Pure cosmetic; does not affect timing of "
            "the underlying motion measurement."
        ),
    )
    return parser


# ----------------------------------------------------------------------------
# Camera / serial helpers
# ----------------------------------------------------------------------------


def _open_serial(port: str, baud: int):
    import serial

    return serial.Serial(port, baud, timeout=1)


def _warm_up_camera(capture, num_frames: int = 5) -> None:
    for _ in range(num_frames):
        ok, _frame = capture.read()
        if not ok:
            break


def _drain_camera_buffer(capture, n: int = CAMERA_BUFFER_DRAIN) -> None:
    """Discard up to n queued frames so the next read() returns a fresh one.

    USB webcams typically buffer ~5 frames; without this drain, read() returns
    the OLDEST queued frame and we end up measuring stale poses (which silently
    looks like 'the car never moved').
    """
    for _ in range(n):
        if not capture.grab():
            break


def _grab_resized_frame(capture) -> np.ndarray:
    """Grab one frame WITHOUT draining (used by GUI live loop)."""
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read a frame from the webcam.")
    return _resize_to_map(frame)


def _grab_fresh_resized_frame(capture) -> np.ndarray:
    """Drain the buffer, then return a freshly grabbed, resized frame."""
    _drain_camera_buffer(capture)
    return _grab_resized_frame(capture)


def _detect_raw(capture) -> dict:
    """One-shot raw detection on a freshly grabbed (drained), resized frame.

    No DetectionSmoother on purpose: the smoother forces the corners onto a
    perfect square and would hide the real rectangular footprint."""
    frame = _grab_fresh_resized_frame(capture)
    det, _body = detect_physical_car(frame)
    return det


def _detect_raw_with_frame(capture) -> tuple[dict, dict, np.ndarray]:
    """Same as _detect_raw but also returns body and the original resized frame.

    Useful for GUI annotation, which needs both the detection dict and the
    body (for the hull outline) plus the underlying frame to draw on.
    """
    frame = _grab_fresh_resized_frame(capture)
    det, body = detect_physical_car(frame)
    return det, body, frame


def _detect_with_smoother(
    capture,
    smoother: DetectionSmoother,
) -> tuple[dict, dict, dict, np.ndarray]:
    """Drain, grab, run raw detection, push it through the smoother.

    Returns (raw_det, smoothed_det, body, frame). Use this at MEASUREMENT
    points (t0 / t1 reads, warmup loop) where you want the freshest possible
    frame. For continuous in-motion feeding use _detect_with_smoother_nodrain
    instead, which avoids burning ~10 frames per call.
    """
    frame = _grab_fresh_resized_frame(capture)
    raw_det, body = detect_physical_car(frame)
    smoothed_det = smoother.update(raw_det)
    return raw_det, smoothed_det, body, frame


def _detect_with_smoother_nodrain(
    capture,
    smoother: DetectionSmoother,
) -> tuple[dict, dict, dict, np.ndarray]:
    """Same as _detect_with_smoother but does NOT drain the camera buffer.

    Used for in-motion smoother feeding where the priority is "keep the
    EMA updated at ~20 Hz" rather than "freshest possible frame." Draining
    every iteration would burn ~10 frames per call and could block on cameras
    that don't honor CAP_PROP_BUFFERSIZE=1.
    """
    frame = _grab_resized_frame(capture)
    raw_det, body = detect_physical_car(frame)
    smoothed_det = smoother.update(raw_det)
    return raw_det, smoothed_det, body, frame


def _warm_smoother(
    capture,
    smoother: DetectionSmoother,
    n_frames: int,
) -> int:
    """Feed n_frames consecutive fresh detections into the smoother.

    Returns the number of successful updates. Detection failures are logged
    once but otherwise tolerated. The smoother's EMA + 4-degree deadband
    needs ~6-10 updates to converge to a new heading after a large jump
    (e.g. post-rotation), so we warm it up explicitly before each measurement.
    """
    successes = 0
    failures = 0
    for _ in range(max(0, n_frames)):
        try:
            frame = _grab_fresh_resized_frame(capture)
            raw_det, _body = detect_physical_car(frame)
            smoother.update(raw_det)
            successes += 1
        except (RuntimeError, ValueError):
            failures += 1
    if failures:
        print(
            f"    [smoother warmup] {failures}/{n_frames} frame(s) failed "
            f"detection; smoother used remaining {successes} sample(s)."
        )
    return successes


def _send(ser, action: str, no_serial: bool) -> None:
    if action not in COMMANDS:
        raise ValueError(f"Unknown action id: {action}")
    if no_serial or ser is None:
        print(f"  [mock] {COMMANDS[action].strip()}")
        return
    send_command(ser, action)


def _countdown(
    seconds: float,
    capture=None,
    gui: bool = False,
    smoother: DetectionSmoother | None = None,
) -> None:
    if seconds <= 0:
        return
    remaining = int(math.ceil(seconds))
    while remaining > 0:
        print(f"  ... next phase in {remaining}s", flush=True)
        if gui and capture is not None:
            try:
                _gui_show_live(
                    capture,
                    1.0,
                    [f"COUNTDOWN: next phase in {remaining}s"],
                    smoother=smoother,
                )
            except QuitRequested:
                raise
        else:
            time.sleep(1.0)
        remaining -= 1


# ----------------------------------------------------------------------------
# GUI helpers
# ----------------------------------------------------------------------------


class QuitRequested(Exception):
    """Raised when the user presses 'q' in the GUI window."""


def _gui_open() -> None:
    cv2.namedWindow(GUI_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(GUI_WINDOW_NAME, 900, 600)


def _gui_close() -> None:
    try:
        cv2.destroyWindow(GUI_WINDOW_NAME)
    except cv2.error:
        pass
    # Some OpenCV builds need a few extra waitKey calls to actually close.
    for _ in range(3):
        cv2.waitKey(1)


def _draw_hud_lines(view: np.ndarray, lines: list[str], origin=(12, 80)) -> None:
    x, y = origin
    for line in lines:
        cv2.rectangle(
            view,
            (x - 4, y - 16),
            (x + 8 + int(7.0 * len(line)), y + 6),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            view, line, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        y += 22


def _draw_smoothed_arrow(
    view: np.ndarray,
    raw_det: dict,
    smoothed_det: dict,
    color: tuple[int, int, int] = (0, 255, 255),
) -> None:
    """Draw the smoother's forward direction as an extra arrow on top of an
    already-annotated view. Color defaults to yellow (BGR (0,255,255)).

    The arrow is anchored at the raw car center (so you can compare it
    visually to the blue raw arrow that annotate_detection draws), and uses
    the smoothed forward_vector for its direction. Length matches the raw
    front-midpoint distance for visual parity.
    """
    cx, cy = raw_det["car_center"]
    fmx, fmy = raw_det["car_direction"]["front_midpoint"]
    arrow_len = float(np.hypot(fmx - cx, fmy - cy))
    if arrow_len < 1.0:
        arrow_len = 60.0
    sf = smoothed_det["car_direction"]["forward_vector"]
    end = (
        int(round(cx + sf[0] * arrow_len)),
        int(round(cy + sf[1] * arrow_len)),
    )
    start = (int(round(cx)), int(round(cy)))
    cv2.arrowedLine(view, start, end, color, 3, tipLength=0.25)


def _annotate_or_failed(
    frame: np.ndarray,
    smoother: DetectionSmoother | None = None,
) -> np.ndarray:
    """Try to detect and annotate; on failure return frame with a FAILED HUD.

    If smoother is provided, also feed it the raw detection AND draw the
    smoothed forward direction as a yellow arrow on top of the annotation.
    """
    try:
        raw_det, body = detect_physical_car(frame)
    except (RuntimeError, ValueError) as exc:
        view = frame.copy()
        cv2.putText(
            view, "DETECTION FAILED", (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            view, str(exc), (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
        )
        return view

    view = annotate_detection(frame, raw_det, body)
    if smoother is not None:
        smoothed_det = smoother.update(raw_det)
        _draw_smoothed_arrow(view, raw_det, smoothed_det)
    return view


def _gui_show_live(
    capture,
    duration_s: float,
    hud_lines: list[str],
    smoother: DetectionSmoother | None = None,
) -> None:
    """Run a live annotated feed for duration_s seconds. Raises QuitRequested on 'q'.

    If smoother is provided, every frame is also pushed through the smoother
    (so its EMA can converge naturally during the live feed) and the smoothed
    forward direction is drawn as a yellow arrow on top of the raw annotation.
    """
    end_t = time.perf_counter() + max(0.0, duration_s)
    while time.perf_counter() < end_t:
        try:
            frame = _grab_resized_frame(capture)
        except RuntimeError:
            if (cv2.waitKey(15) & 0xFF) == ord("q"):
                raise QuitRequested()
            continue
        view = _annotate_or_failed(frame, smoother=smoother)
        _draw_hud_lines(view, hud_lines)
        cv2.imshow(GUI_WINDOW_NAME, view)
        if (cv2.waitKey(15) & 0xFF) == ord("q"):
            raise QuitRequested()


def _gui_freeze(view: np.ndarray, duration_s: float) -> None:
    """Hold a single frozen view for duration_s seconds. Raises QuitRequested on 'q'."""
    end_t = time.perf_counter() + max(0.0, duration_s)
    while time.perf_counter() < end_t:
        cv2.imshow(GUI_WINDOW_NAME, view)
        if (cv2.waitKey(15) & 0xFF) == ord("q"):
            raise QuitRequested()


def _build_comparison_view(
    frame_t0: np.ndarray,
    det_t0: dict,
    body_t0: dict,
    frame_t1: np.ndarray,
    det_t1: dict,
    body_t1: dict,
    hud_lines: list[str],
    smoothed_forward_t0: np.ndarray | None = None,
    smoothed_forward_t1: np.ndarray | None = None,
) -> np.ndarray:
    """Render a single 900x600 view with both t0 (cyan) and t1 (yellow) overlays
    on top of the t1 frame, plus a magenta arrow for the center displacement.

    If smoothed_forward_t0 / smoothed_forward_t1 are provided, the smoothed
    direction at each time is drawn as a thicker yellow arrow on top of the
    raw arrows, so the user can see exactly which forward vector the speed
    measurement is using.
    """
    view = annotate_detection(frame_t1, det_t1, body_t1)

    # Overlay the t0 polygon (cyan) so you can see where the car was BEFORE.
    c0 = det_t0["car_corners"]
    poly0 = np.array(
        [c0["front_right"], c0["front_left"],
         c0["back_left"],   c0["back_right"]],
        dtype=np.int32,
    )
    cv2.polylines(view, [poly0], isClosed=True, color=(255, 255, 0), thickness=2)

    # t0 RAW heading arrow in cyan.
    ctr0 = tuple(map(int, det_t0["car_center"]))
    tip0 = tuple(map(int, det_t0["car_direction"]["front_midpoint"]))
    cv2.circle(view, ctr0, 4, (255, 255, 0), -1)
    cv2.arrowedLine(view, ctr0, tip0, (255, 255, 0), 2, tipLength=0.25)

    # t0 SMOOTHED heading arrow in yellow (the direction we measure with).
    if smoothed_forward_t0 is not None:
        cx0, cy0 = det_t0["car_center"]
        fmx0, fmy0 = det_t0["car_direction"]["front_midpoint"]
        arrow_len_0 = max(60.0, float(np.hypot(fmx0 - cx0, fmy0 - cy0)))
        end0 = (
            int(round(cx0 + smoothed_forward_t0[0] * arrow_len_0)),
            int(round(cy0 + smoothed_forward_t0[1] * arrow_len_0)),
        )
        cv2.arrowedLine(view, ctr0, end0, (0, 255, 255), 3, tipLength=0.25)

    # t1 SMOOTHED heading arrow in yellow.
    if smoothed_forward_t1 is not None:
        cx1, cy1 = det_t1["car_center"]
        fmx1, fmy1 = det_t1["car_direction"]["front_midpoint"]
        arrow_len_1 = max(60.0, float(np.hypot(fmx1 - cx1, fmy1 - cy1)))
        end1 = (
            int(round(cx1 + smoothed_forward_t1[0] * arrow_len_1)),
            int(round(cy1 + smoothed_forward_t1[1] * arrow_len_1)),
        )
        cv2.arrowedLine(
            view,
            (int(round(cx1)), int(round(cy1))),
            end1,
            (0, 255, 255), 3, tipLength=0.25,
        )

    # Magenta center-to-center displacement arrow.
    ctr1 = tuple(map(int, det_t1["car_center"]))
    cv2.arrowedLine(view, ctr0, ctr1, (255, 0, 255), 2, tipLength=0.2)

    # Legend.
    legend = [
        "cyan polygon + cyan arrow = t0 (pre-motion, raw)",
        "green polygon + blue arrow = t1 (post-motion, raw)",
        "yellow arrow = SMOOTHED forward (used for measurements)",
        "magenta arrow = center displacement",
    ]
    _draw_hud_lines(view, legend + [""] + hud_lines, origin=(12, 100))
    return view


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------


def _corner_array(det: dict, key: str) -> np.ndarray:
    return np.asarray(det["car_corners"][key], dtype=np.float64)


def _footprint_dims(det: dict) -> tuple[float, float]:
    """Return (length_px, width_px) from the labeled corners of one detection."""
    fr = _corner_array(det, "front_right")
    fl = _corner_array(det, "front_left")
    br = _corner_array(det, "back_right")
    bl = _corner_array(det, "back_left")

    front_mid = 0.5 * (fr + fl)
    back_mid = 0.5 * (br + bl)

    length_px = float(np.linalg.norm(front_mid - back_mid))
    width_px = float(np.linalg.norm(fr - fl))
    return length_px, width_px


def _signed_forward_displacement(
    center_t0: np.ndarray,
    forward_t0: np.ndarray,
    center_t1: np.ndarray,
) -> tuple[float, float]:
    """Return (signed_forward_px, total_displacement_px).

    Signed projection onto the pre-motion forward axis: positive means the car
    moved in the direction it was facing at t0.
    """
    delta = center_t1 - center_t0
    total = float(np.linalg.norm(delta))
    fwd_norm = float(np.linalg.norm(forward_t0))
    if fwd_norm < 1e-9:
        return 0.0, total
    forward_unit = forward_t0 / fwd_norm
    signed = float(np.dot(delta, forward_unit))
    return signed, total


def _signed_heading_delta_deg(forward_t0: np.ndarray, forward_t1: np.ndarray) -> float:
    """Signed angle from forward_t0 to forward_t1 in degrees, in (-180, 180].

    Uses atan2(cross, dot) so 0/360 wrap-around cannot break the result. Sign
    convention matches the simulator: positive => CCW in the image-y-down
    frame (matches TestRig.Car: action 2 increases self.angle => CCW).
    """
    n0 = float(np.linalg.norm(forward_t0))
    n1 = float(np.linalg.norm(forward_t1))
    if n0 < 1e-9 or n1 < 1e-9:
        return 0.0
    a = forward_t0 / n0
    b = forward_t1 / n1

    dot = float(np.dot(a, b))
    # 2D "cross" (signed scalar). With image-y-down, positive cross product
    # of (a, b) means b is CCW of a, which matches sim action-2 sign.
    cross = float(a[0] * b[1] - a[1] * b[0])
    # Negate so that CCW (heading increasing in TestRig's convention) is +.
    delta_rad = math.atan2(-cross, dot)
    return math.degrees(delta_rad)


# ----------------------------------------------------------------------------
# Stats helper
# ----------------------------------------------------------------------------


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": float("nan"), "stdev": float("nan"),
                "min": float("nan"), "max": float("nan")}
    n = len(values)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "stdev": stdev,
        "min": min(values),
        "max": max(values),
    }


# ----------------------------------------------------------------------------
# Phase implementations
# ----------------------------------------------------------------------------


def measure_footprint(
    capture,
    num_frames: int,
    gui: bool,
    smoother: DetectionSmoother,
) -> dict:
    print(f"\n=== Phase 1: Footprint ({num_frames} frames, raw detector) ===")
    print("  Hold the car still. Sampling now ...")

    lengths: list[float] = []
    widths: list[float] = []
    failures = 0

    for i in range(num_frames):
        try:
            raw_det, smoothed_det, body, frame = _detect_with_smoother(
                capture, smoother
            )
        except (RuntimeError, ValueError) as exc:
            failures += 1
            print(f"  [frame {i + 1}/{num_frames}] detection failed: {exc}")
            if gui:
                # Still show the camera so the user can see what's wrong.
                try:
                    raw = _grab_resized_frame(capture)
                    view = raw.copy()
                    cv2.putText(
                        view, "DETECTION FAILED", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(GUI_WINDOW_NAME, view)
                    if (cv2.waitKey(15) & 0xFF) == ord("q"):
                        raise QuitRequested()
                except RuntimeError:
                    pass
            continue
        length_px, width_px = _footprint_dims(raw_det)
        lengths.append(length_px)
        widths.append(width_px)

        if gui:
            view = annotate_detection(frame, raw_det, body)
            _draw_smoothed_arrow(view, raw_det, smoothed_det)
            _draw_hud_lines(
                view,
                [
                    f"PHASE 1: Footprint ({i + 1}/{num_frames})",
                    f"length_px = {length_px:6.2f}",
                    f"width_px  = {width_px:6.2f}",
                    "Yellow arrow = smoothed forward (measurements use this)",
                    "Hold the car still.",
                ],
            )
            cv2.imshow(GUI_WINDOW_NAME, view)
            if (cv2.waitKey(15) & 0xFF) == ord("q"):
                raise QuitRequested()

    print(
        f"  collected {len(lengths)} good samples, {failures} failures "
        f"(of {num_frames} requested)"
    )
    return {
        "length": _summarize(lengths),
        "width": _summarize(widths),
        "samples": len(lengths),
        "failures": failures,
    }


def _run_motion_trial(
    capture,
    ser,
    no_serial: bool,
    smoother: DetectionSmoother,
    action: str,
    motion_duration: float,
    settle_duration: float,
    gui: bool,
    gui_freeze: float,
    phase_label: str,
    trial_idx: int,
    trial_total: int,
) -> dict | None:
    """Run a single motion trial. Returns per-trial measurements or None on failure.

    Direction (forward_vector) at both t0 and t1 comes from the SMOOTHER, not
    from a single raw detection, so frame-to-frame white-tape jitter cannot
    poison the heading. The smoother is warmed up immediately before each
    capture point and continuously fed during the motion + settle window, so
    its EMA actually has time to converge.

    Center stays raw (the centroid of the big red blob is stable enough on
    its own). Footprint corners are unchanged elsewhere.
    """
    # ---- Pre-motion warmup --------------------------------------------------
    # Bring the smoother's forward in line with the car's CURRENT pre-motion
    # direction (it may have lagged from the previous trial / phase).
    _warm_smoother(capture, smoother, SMOOTHER_PRE_MOTION_WARMUP_FRAMES)

    try:
        raw_det0, smoothed_det0, body0, frame0 = _detect_with_smoother(
            capture, smoother
        )
    except (RuntimeError, ValueError) as exc:
        print(f"    pre-motion detection failed: {exc}")
        # Best-effort stop in case the previous trial left the motors live.
        try:
            _send(ser, "4", no_serial)
        except Exception:
            pass
        return None

    center_t0 = np.asarray(raw_det0["car_center"], dtype=np.float64)
    raw_forward_t0 = np.asarray(
        raw_det0["car_direction"]["forward_vector"], dtype=np.float64
    )
    forward_t0 = np.asarray(
        smoothed_det0["car_direction"]["forward_vector"], dtype=np.float64
    )

    if gui:
        view = annotate_detection(frame0, raw_det0, body0)
        _draw_smoothed_arrow(view, raw_det0, smoothed_det0)
        _draw_hud_lines(
            view,
            [
                f"{phase_label}  trial {trial_idx}/{trial_total}",
                "T0 (pre-motion) - about to send command",
                f"action {action} -> '{COMMANDS[action].strip()}'",
                f"center_t0      = ({center_t0[0]:6.1f}, {center_t0[1]:6.1f})",
                f"raw_forward_t0 = ({raw_forward_t0[0]:+.3f}, {raw_forward_t0[1]:+.3f})",
                f"smo_forward_t0 = ({forward_t0[0]:+.3f}, {forward_t0[1]:+.3f})",
            ],
        )
        _gui_freeze(view, gui_freeze)

    # ---- Drive --------------------------------------------------------------
    t_start = time.perf_counter()
    _send(ser, action, no_serial)

    if gui:
        # Live annotated feed while the car is supposed to be moving. The
        # smoother is fed every frame so its EMA tracks the rotation/translation.
        live_hud = [
            f"{phase_label}  trial {trial_idx}/{trial_total}",
            "MOVING - command in flight",
            f"action {action}",
        ]
        try:
            _gui_show_live(capture, motion_duration, live_hud, smoother=smoother)
        except QuitRequested:
            _send(ser, "4", no_serial)
            raise
    else:
        # Headless: sleep loop, but periodically detect+update the smoother so
        # it actually sees the rotation in flight (otherwise the post-motion
        # frame would be a single 100+ degree jump that the smoother's EMA
        # would only partially absorb in one update).
        last_feed = 0.0
        while time.perf_counter() - t_start < motion_duration:
            now = time.perf_counter()
            if now - last_feed >= HEADLESS_MOTION_FEED_INTERVAL_S:
                try:
                    _detect_with_smoother_nodrain(capture, smoother)
                except (RuntimeError, ValueError):
                    # Detection mid-motion can occasionally fail (motion blur,
                    # tape partially occluded by hand, etc). Keep going.
                    pass
                last_feed = now
            else:
                time.sleep(0.005)

    _send(ser, "4", no_serial)
    elapsed = time.perf_counter() - t_start

    # ---- Settle -------------------------------------------------------------
    if gui:
        try:
            _gui_show_live(
                capture,
                settle_duration,
                [f"{phase_label}  trial {trial_idx}/{trial_total}", "SETTLING ..."],
                smoother=smoother,
            )
        except QuitRequested:
            raise
    else:
        # Keep feeding the smoother during settle too.
        settle_end = time.perf_counter() + settle_duration
        last_feed = 0.0
        while time.perf_counter() < settle_end:
            now = time.perf_counter()
            if now - last_feed >= HEADLESS_MOTION_FEED_INTERVAL_S:
                try:
                    _detect_with_smoother_nodrain(capture, smoother)
                except (RuntimeError, ValueError):
                    pass
                last_feed = now
            else:
                time.sleep(0.005)

    # ---- Post-motion warmup -------------------------------------------------
    # Push the smoother the rest of the way to the post-motion direction.
    _warm_smoother(capture, smoother, SMOOTHER_POST_MOTION_WARMUP_FRAMES)

    try:
        raw_det1, smoothed_det1, body1, frame1 = _detect_with_smoother(
            capture, smoother
        )
    except (RuntimeError, ValueError) as exc:
        print(f"    post-motion detection failed: {exc}")
        return None

    center_t1 = np.asarray(raw_det1["car_center"], dtype=np.float64)
    raw_forward_t1 = np.asarray(
        raw_det1["car_direction"]["forward_vector"], dtype=np.float64
    )
    forward_t1 = np.asarray(
        smoothed_det1["car_direction"]["forward_vector"], dtype=np.float64
    )

    signed_fwd_px, total_disp_px = _signed_forward_displacement(
        center_t0, forward_t0, center_t1
    )
    heading_delta_deg = _signed_heading_delta_deg(forward_t0, forward_t1)
    raw_heading_delta_deg = _signed_heading_delta_deg(raw_forward_t0, raw_forward_t1)

    if gui:
        comparison_hud = [
            f"{phase_label}  trial {trial_idx}/{trial_total}  (T1)",
            f"elapsed             = {elapsed:.3f}s",
            f"center_t1           = ({center_t1[0]:6.1f}, {center_t1[1]:6.1f})",
            f"|disp|              = {total_disp_px:6.2f} px",
            f"signed_fwd          = {signed_fwd_px:+6.2f} px",
            f"d_heading (smoothed)= {heading_delta_deg:+6.2f} deg",
            f"d_heading (raw)     = {raw_heading_delta_deg:+6.2f} deg",
        ]
        view = _build_comparison_view(
            frame0, raw_det0, body0,
            frame1, raw_det1, body1,
            comparison_hud,
            smoothed_forward_t0=forward_t0,
            smoothed_forward_t1=forward_t1,
        )
        _gui_freeze(view, gui_freeze)

    return {
        "elapsed": elapsed,
        "signed_fwd_px": signed_fwd_px,
        "total_disp_px": total_disp_px,
        "heading_delta_deg": heading_delta_deg,
        "raw_heading_delta_deg": raw_heading_delta_deg,
    }


def measure_linear(
    capture,
    ser,
    no_serial: bool,
    smoother: DetectionSmoother,
    action: str,
    label: str,
    trials: int,
    motion_duration: float,
    settle_duration: float,
    gui: bool,
    gui_freeze: float,
) -> dict:
    print(f"\n=== Phase: {label} (action {action}, trials={trials}) ===")
    speeds: list[float] = []
    failures = 0

    for trial in range(1, trials + 1):
        print(f"  trial {trial}/{trials} ...")
        result = _run_motion_trial(
            capture,
            ser,
            no_serial,
            smoother,
            action,
            motion_duration,
            settle_duration,
            gui=gui,
            gui_freeze=gui_freeze,
            phase_label=label,
            trial_idx=trial,
            trial_total=trials,
        )
        if result is None:
            failures += 1
            continue

        speed_px_s = result["signed_fwd_px"] / result["elapsed"]
        speeds.append(speed_px_s)
        print(
            f"    elapsed={result['elapsed']:.3f}s  "
            f"signed_fwd={result['signed_fwd_px']:+7.2f}px  "
            f"|disp|={result['total_disp_px']:6.2f}px  "
            f"=> {speed_px_s:+7.2f} px/s"
        )

    return {
        "speed_px_s": _summarize(speeds),
        "failures": failures,
    }


def measure_angular(
    capture,
    ser,
    no_serial: bool,
    smoother: DetectionSmoother,
    action: str,
    label: str,
    trials: int,
    motion_duration: float,
    settle_duration: float,
    gui: bool,
    gui_freeze: float,
) -> dict:
    print(f"\n=== Phase: {label} (action {action}, trials={trials}) ===")
    rates: list[float] = []
    raw_rates: list[float] = []
    drifts: list[float] = []
    failures = 0

    for trial in range(1, trials + 1):
        print(f"  trial {trial}/{trials} ...")
        result = _run_motion_trial(
            capture,
            ser,
            no_serial,
            smoother,
            action,
            motion_duration,
            settle_duration,
            gui=gui,
            gui_freeze=gui_freeze,
            phase_label=label,
            trial_idx=trial,
            trial_total=trials,
        )
        if result is None:
            failures += 1
            continue

        rate_deg_s = result["heading_delta_deg"] / result["elapsed"]
        raw_rate_deg_s = result["raw_heading_delta_deg"] / result["elapsed"]
        rates.append(rate_deg_s)
        raw_rates.append(raw_rate_deg_s)
        drifts.append(result["total_disp_px"])
        print(
            f"    elapsed={result['elapsed']:.3f}s  "
            f"d_heading(smoothed)={result['heading_delta_deg']:+7.2f}deg  "
            f"d_heading(raw)={result['raw_heading_delta_deg']:+7.2f}deg  "
            f"drift={result['total_disp_px']:6.2f}px  "
            f"=> {rate_deg_s:+7.2f} deg/s (smoothed)"
        )

    return {
        "rate_deg_s": _summarize(rates),
        "raw_rate_deg_s": _summarize(raw_rates),
        "drift_px": _summarize(drifts),
        "failures": failures,
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


def _fmt_summary(s: dict, unit: str) -> str:
    if s["n"] == 0:
        return f"  no valid samples"
    return (
        f"  mean={s['mean']:+8.2f} {unit}   stdev={s['stdev']:6.2f}   "
        f"min={s['min']:+8.2f}   max={s['max']:+8.2f}   n={s['n']}"
    )


def print_report(
    footprint: dict,
    forward: dict,
    backward: dict,
    ccw: dict,
    cw: dict,
    args: argparse.Namespace,
) -> None:
    print("\n" + "=" * 72)
    print("CALIBRATION REPORT")
    print("=" * 72)

    print(
        f"\n=== Car footprint (resized 900x600 frame, raw detector, "
        f"N={footprint['samples']}) ==="
    )
    print(
        f"  length_px : mean={footprint['length']['mean']:6.2f}  "
        f"stdev={footprint['length']['stdev']:5.2f}   "
        f"(sim CAR_LENGTH = {SIM_CAR_LENGTH})"
    )
    print(
        f"  width_px  : mean={footprint['width']['mean']:6.2f}  "
        f"stdev={footprint['width']['stdev']:5.2f}   "
        f"(sim CAR_WIDTH  = {SIM_CAR_WIDTH})"
    )
    if footprint["failures"]:
        print(f"  ({footprint['failures']} frame(s) failed detection)")

    print(
        f"\n=== Linear speed (motion_duration={args.linear_motion_duration}s, "
        f"trials={args.trials}) ==="
    )
    print("  action 0 (forward) :")
    print(_fmt_summary(forward["speed_px_s"], "px/s"))
    if forward["failures"]:
        print(f"    ({forward['failures']} trial(s) failed)")
    print("  action 1 (backward):")
    print(_fmt_summary(backward["speed_px_s"], "px/s"))
    if backward["failures"]:
        print(f"    ({backward['failures']} trial(s) failed)")
    print(f"  (sim CAR_SPEED = {SIM_CAR_SPEED} px/s)")

    print(
        f"\n=== Angular speed (motion_duration={args.angular_motion_duration}s, "
        f"trials={args.trials}) ==="
    )
    print("  action 2 (rotate left / CCW) :")
    print("    smoothed:", _fmt_summary(ccw["rate_deg_s"], "deg/s").strip())
    print("    raw     :", _fmt_summary(ccw["raw_rate_deg_s"], "deg/s").strip())
    if ccw["failures"]:
        print(f"    ({ccw['failures']} trial(s) failed)")
    print("  action 3 (rotate right / CW) :")
    print("    smoothed:", _fmt_summary(cw["rate_deg_s"], "deg/s").strip())
    print("    raw     :", _fmt_summary(cw["raw_rate_deg_s"], "deg/s").strip())
    if cw["failures"]:
        print(f"    ({cw['failures']} trial(s) failed)")
    print(
        f"  (sim CAR_ROT_SPEED = {SIM_CAR_ROT_SPEED} deg/s; sign convention: + = CCW)"
    )
    print(
        "  ('smoothed' uses DetectionSmoother forward; 'raw' uses one-shot "
        "tape-based forward. Big disagreement = noisy white tape; trust smoothed.)"
    )

    print("\n  drift during rotation (total displacement, |delta_center|):")
    print(f"    action 2 (CCW): {_fmt_summary(ccw['drift_px'], 'px').strip()}")
    print(f"    action 3 (CW) : {_fmt_summary(cw['drift_px'], 'px').strip()}")
    print(
        "  (non-trivial drift indicates the real car does not pivot in place; "
        "useful when extending the simulator)"
    )
    print()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    print("[calibrate_car] Opening camera ...")
    capture = _open_camera(args.camera_index)
    _configure_capture(capture, 1920, 1080)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")
    # Best-effort: shrink the driver-side queue to 1 so cv2.read() returns
    # the latest frame instead of the oldest queued one. Some backends
    # silently ignore this; we still drain explicitly before each measurement.
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    _warm_up_camera(capture)
    print(f"[calibrate_car] Camera ready (index {args.camera_index}).")

    ser = None
    if not args.no_serial:
        print(f"[calibrate_car] Opening serial port {args.port} @ {args.baud} ...")
        ser = _open_serial(args.port, args.baud)
        print("[calibrate_car] Serial connected.")
    else:
        print("[calibrate_car] --no-serial set; commands will be printed only.")

    if args.gui:
        _gui_open()
        print(
            "[calibrate_car] GUI mode: a window has opened. Press 'q' inside "
            "that window at any time to quit early."
        )

    # If the user passed the legacy --motion-duration override, it wins for
    # both linear and angular phases; otherwise each phase uses its own flag.
    if args.motion_duration is not None:
        linear_dur = args.motion_duration
        angular_dur = args.motion_duration
    else:
        linear_dur = args.linear_motion_duration
        angular_dur = args.angular_motion_duration
    args.linear_motion_duration = linear_dur
    args.angular_motion_duration = angular_dur

    # One persistent smoother across the whole run. Direction at every t0/t1
    # measurement read uses smoother.update(...) output; the smoother is fed
    # continuously (footprint loop, GUI live feed, headless motion drain) so
    # its EMA actually has time to converge.
    smoother = DetectionSmoother()

    footprint = forward = backward = ccw = cw = None
    quit_early = False
    try:
        footprint = measure_footprint(
            capture, args.footprint_frames, gui=args.gui, smoother=smoother
        )

        _countdown(args.countdown, capture=capture, gui=args.gui, smoother=smoother)
        forward = measure_linear(
            capture, ser, args.no_serial, smoother,
            action="0", label="Forward speed",
            trials=args.trials,
            motion_duration=linear_dur,
            settle_duration=args.settle_duration,
            gui=args.gui, gui_freeze=args.gui_freeze,
        )

        _countdown(args.countdown, capture=capture, gui=args.gui, smoother=smoother)
        backward = measure_linear(
            capture, ser, args.no_serial, smoother,
            action="1", label="Backward speed",
            trials=args.trials,
            motion_duration=linear_dur,
            settle_duration=args.settle_duration,
            gui=args.gui, gui_freeze=args.gui_freeze,
        )

        _countdown(args.countdown, capture=capture, gui=args.gui, smoother=smoother)
        ccw = measure_angular(
            capture, ser, args.no_serial, smoother,
            action="2", label="Rotate left (CCW)",
            trials=args.trials,
            motion_duration=angular_dur,
            settle_duration=args.settle_duration,
            gui=args.gui, gui_freeze=args.gui_freeze,
        )

        _countdown(args.countdown, capture=capture, gui=args.gui, smoother=smoother)
        cw = measure_angular(
            capture, ser, args.no_serial, smoother,
            action="3", label="Rotate right (CW)",
            trials=args.trials,
            motion_duration=angular_dur,
            settle_duration=args.settle_duration,
            gui=args.gui, gui_freeze=args.gui_freeze,
        )

    except QuitRequested:
        quit_early = True
        print("[calibrate_car] Quit requested via 'q'; stopping early.")
    finally:
        # Always send a stop, regardless of how we got here.
        try:
            _send(ser, "4", args.no_serial)
        except Exception as exc:
            print(f"[calibrate_car] Warning: failed to send stop command: {exc}")

        if ser is not None:
            try:
                if ser.is_open:
                    ser.close()
                    print("[calibrate_car] Serial disconnected.")
            except Exception as exc:
                print(f"[calibrate_car] Warning: failed to close serial: {exc}")

        try:
            capture.release()
            print("[calibrate_car] Camera released.")
        except Exception as exc:
            print(f"[calibrate_car] Warning: failed to release camera: {exc}")

        if args.gui:
            _gui_close()
            print("[calibrate_car] GUI window closed.")

    if all(p is not None for p in (footprint, forward, backward, ccw, cw)):
        print_report(footprint, forward, backward, ccw, cw, args)
    elif quit_early:
        print(
            "[calibrate_car] Early quit requested; the phases that did "
            "complete are printed above. No consolidated report."
        )
    else:
        print(
            "[calibrate_car] One or more phases were skipped due to an early "
            "exit; no consolidated report will be printed."
        )


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

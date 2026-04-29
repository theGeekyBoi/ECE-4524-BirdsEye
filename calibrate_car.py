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
Bluetooth serial path, sleep --motion-duration seconds, send action 4 (stop),
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
    _configure_capture,
    _open_camera,
    annotate_detection,
    detect_physical_car,
)
from Target_Tracking.state_generator import _resize_to_map  # noqa: E402
from Bluetooth_Comms.remote_control import COMMANDS, PORT, send_command  # noqa: E402

import TestRig  # noqa: E402  (only used to read sim constants for the report)


GUI_WINDOW_NAME = "calibrate_car (GUI)"
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
        "--motion-duration",
        type=float,
        default=1.0,
        help="Seconds to hold the action during each motion trial (default: 1.0).",
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


def _send(ser, action: str, no_serial: bool) -> None:
    if action not in COMMANDS:
        raise ValueError(f"Unknown action id: {action}")
    if no_serial or ser is None:
        print(f"  [mock] {COMMANDS[action].strip()}")
        return
    send_command(ser, action)


def _countdown(seconds: float, capture=None, gui: bool = False) -> None:
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


def _annotate_or_failed(frame: np.ndarray) -> np.ndarray:
    """Try to detect and annotate; on failure return frame with a FAILED HUD."""
    try:
        det, body = detect_physical_car(frame)
        return annotate_detection(frame, det, body)
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


def _gui_show_live(
    capture,
    duration_s: float,
    hud_lines: list[str],
) -> None:
    """Run a live annotated feed for duration_s seconds. Raises QuitRequested on 'q'."""
    end_t = time.perf_counter() + max(0.0, duration_s)
    while time.perf_counter() < end_t:
        try:
            frame = _grab_resized_frame(capture)
        except RuntimeError:
            if (cv2.waitKey(15) & 0xFF) == ord("q"):
                raise QuitRequested()
            continue
        view = _annotate_or_failed(frame)
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
) -> np.ndarray:
    """Render a single 900x600 view with both t0 (cyan) and t1 (yellow) overlays
    on top of the t1 frame, plus a magenta arrow for the center displacement.

    This is the "did the car actually move?" frame the user can stare at to
    diagnose stale-buffer or detector-lock issues.
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

    # t0 heading arrow in cyan.
    ctr0 = tuple(map(int, det_t0["car_center"]))
    tip0 = tuple(map(int, det_t0["car_direction"]["front_midpoint"]))
    cv2.circle(view, ctr0, 4, (255, 255, 0), -1)
    cv2.arrowedLine(view, ctr0, tip0, (255, 255, 0), 2, tipLength=0.25)

    # Magenta center-to-center displacement arrow.
    ctr1 = tuple(map(int, det_t1["car_center"]))
    cv2.arrowedLine(view, ctr0, ctr1, (255, 0, 255), 2, tipLength=0.2)

    # Legend.
    legend = [
        "cyan polygon/arrow = t0 (pre-motion)",
        "green polygon + blue arrow = t1 (post-motion)",
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


def measure_footprint(capture, num_frames: int, gui: bool) -> dict:
    print(f"\n=== Phase 1: Footprint ({num_frames} frames, raw detector) ===")
    print("  Hold the car still. Sampling now ...")

    lengths: list[float] = []
    widths: list[float] = []
    failures = 0

    for i in range(num_frames):
        try:
            det, body, frame = _detect_raw_with_frame(capture)
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
        length_px, width_px = _footprint_dims(det)
        lengths.append(length_px)
        widths.append(width_px)

        if gui:
            view = annotate_detection(frame, det, body)
            _draw_hud_lines(
                view,
                [
                    f"PHASE 1: Footprint ({i + 1}/{num_frames})",
                    f"length_px = {length_px:6.2f}",
                    f"width_px  = {width_px:6.2f}",
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

    In GUI mode also: shows the t0 snapshot frozen for gui_freeze seconds, runs
    a live annotated feed during motion + settle (so the user can visually
    confirm the car actually moves), and finally shows a t0/t1 comparison
    overlay with the displacement vector.
    """
    try:
        det0, body0, frame0 = _detect_raw_with_frame(capture)
    except (RuntimeError, ValueError) as exc:
        print(f"    pre-motion detection failed: {exc}")
        # Best-effort stop in case the previous trial left the motors live.
        try:
            _send(ser, "4", no_serial)
        except Exception:
            pass
        return None

    center_t0 = np.asarray(det0["car_center"], dtype=np.float64)
    forward_t0 = np.asarray(det0["car_direction"]["forward_vector"], dtype=np.float64)

    if gui:
        view = annotate_detection(frame0, det0, body0)
        _draw_hud_lines(
            view,
            [
                f"{phase_label}  trial {trial_idx}/{trial_total}",
                "T0 (pre-motion) - about to send command",
                f"action {action} -> '{COMMANDS[action].strip()}'",
                f"center_t0 = ({center_t0[0]:6.1f}, {center_t0[1]:6.1f})",
                f"forward_t0 = ({forward_t0[0]:+.3f}, {forward_t0[1]:+.3f})",
            ],
        )
        _gui_freeze(view, gui_freeze)

    # Drive.
    t_start = time.perf_counter()
    _send(ser, action, no_serial)

    if gui:
        # Live annotated feed while the car is supposed to be moving. This is
        # the key diagnostic frame: if the car physically moves but the green
        # polygon doesn't track, the detector is locked onto the wrong thing.
        live_hud = [
            f"{phase_label}  trial {trial_idx}/{trial_total}",
            "MOVING - command in flight",
            f"action {action}",
        ]
        try:
            _gui_show_live(capture, motion_duration, live_hud)
        except QuitRequested:
            _send(ser, "4", no_serial)
            raise
    else:
        while time.perf_counter() - t_start < motion_duration:
            # Tight sleep loop - keep the action held for as close to
            # motion_duration as possible without busy-waiting.
            time.sleep(0.005)

    _send(ser, "4", no_serial)
    elapsed = time.perf_counter() - t_start

    # Let the car actually halt before we measure.
    if gui:
        try:
            _gui_show_live(
                capture,
                settle_duration,
                [f"{phase_label}  trial {trial_idx}/{trial_total}", "SETTLING ..."],
            )
        except QuitRequested:
            raise
    else:
        time.sleep(settle_duration)

    try:
        det1, body1, frame1 = _detect_raw_with_frame(capture)
    except (RuntimeError, ValueError) as exc:
        print(f"    post-motion detection failed: {exc}")
        return None

    center_t1 = np.asarray(det1["car_center"], dtype=np.float64)
    forward_t1 = np.asarray(det1["car_direction"]["forward_vector"], dtype=np.float64)

    signed_fwd_px, total_disp_px = _signed_forward_displacement(
        center_t0, forward_t0, center_t1
    )
    heading_delta_deg = _signed_heading_delta_deg(forward_t0, forward_t1)

    if gui:
        comparison_hud = [
            f"{phase_label}  trial {trial_idx}/{trial_total}  (T1)",
            f"elapsed       = {elapsed:.3f}s",
            f"center_t1     = ({center_t1[0]:6.1f}, {center_t1[1]:6.1f})",
            f"|disp|        = {total_disp_px:6.2f} px",
            f"signed_fwd    = {signed_fwd_px:+6.2f} px",
            f"d_heading     = {heading_delta_deg:+6.2f} deg",
        ]
        view = _build_comparison_view(
            frame0, det0, body0, frame1, det1, body1, comparison_hud
        )
        _gui_freeze(view, gui_freeze)

    return {
        "elapsed": elapsed,
        "signed_fwd_px": signed_fwd_px,
        "total_disp_px": total_disp_px,
        "heading_delta_deg": heading_delta_deg,
    }


def measure_linear(
    capture,
    ser,
    no_serial: bool,
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
    drifts: list[float] = []
    failures = 0

    for trial in range(1, trials + 1):
        print(f"  trial {trial}/{trials} ...")
        result = _run_motion_trial(
            capture,
            ser,
            no_serial,
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
        rates.append(rate_deg_s)
        drifts.append(result["total_disp_px"])
        print(
            f"    elapsed={result['elapsed']:.3f}s  "
            f"d_heading={result['heading_delta_deg']:+7.2f}deg  "
            f"drift={result['total_disp_px']:6.2f}px  "
            f"=> {rate_deg_s:+7.2f} deg/s"
        )

    return {
        "rate_deg_s": _summarize(rates),
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
        f"\n=== Linear speed (motion_duration={args.motion_duration}s, "
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
        f"\n=== Angular speed (motion_duration={args.motion_duration}s, "
        f"trials={args.trials}) ==="
    )
    print("  action 2 (rotate left / CCW) :")
    print(_fmt_summary(ccw["rate_deg_s"], "deg/s"))
    if ccw["failures"]:
        print(f"    ({ccw['failures']} trial(s) failed)")
    print("  action 3 (rotate right / CW) :")
    print(_fmt_summary(cw["rate_deg_s"], "deg/s"))
    if cw["failures"]:
        print(f"    ({cw['failures']} trial(s) failed)")
    print(f"  (sim CAR_ROT_SPEED = {SIM_CAR_ROT_SPEED} deg/s; sign convention: + = CCW)")

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

    footprint = forward = backward = ccw = cw = None
    quit_early = False
    try:
        footprint = measure_footprint(capture, args.footprint_frames, gui=args.gui)

        _countdown(args.countdown, capture=capture, gui=args.gui)
        forward = measure_linear(
            capture, ser, args.no_serial,
            action="0", label="Forward speed",
            trials=args.trials,
            motion_duration=args.motion_duration,
            settle_duration=args.settle_duration,
            gui=args.gui, gui_freeze=args.gui_freeze,
        )

        _countdown(args.countdown, capture=capture, gui=args.gui)
        backward = measure_linear(
            capture, ser, args.no_serial,
            action="1", label="Backward speed",
            trials=args.trials,
            motion_duration=args.motion_duration,
            settle_duration=args.settle_duration,
            gui=args.gui, gui_freeze=args.gui_freeze,
        )

        _countdown(args.countdown, capture=capture, gui=args.gui)
        ccw = measure_angular(
            capture, ser, args.no_serial,
            action="2", label="Rotate left (CCW)",
            trials=args.trials,
            motion_duration=args.motion_duration,
            settle_duration=args.settle_duration,
            gui=args.gui, gui_freeze=args.gui_freeze,
        )

        _countdown(args.countdown, capture=capture, gui=args.gui)
        cw = measure_angular(
            capture, ser, args.no_serial,
            action="3", label="Rotate right (CW)",
            trials=args.trials,
            motion_duration=args.motion_duration,
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

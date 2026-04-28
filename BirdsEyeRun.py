"""
BirdsEyeRun.py

Closed-loop runtime that ties together the three BirdsEye subsystems:

    overhead webcam
        -> Target_Tracking detectors (car + target)
        -> 22-d normalized state vector (state_generator._assemble_vector)
        -> trained DQN policy (dqn_agent.DQNAgent + dqn_model.pth)
        -> serial command to the physical car (Bluetooth_Comms.remote_control)

Two modes:
    --mode gui       : default. Opens a debug window showing the car mask,
                       target mask, oriented car polygon, target circle, a
                       heading arrow, and a HUD with the chosen action.
                       Quit with 'q' in that window.
    --mode headless  : no window. Per-frame status line printed to stdout.
                       Quit with 'q' in the terminal (Windows-only via msvcrt).

Behavior:
    - The program never auto-terminates. The only exit path is the user
      pressing 'q'. Detection failures and reaching the target both keep the
      loop alive.
    - When the car is within --reach-threshold pixels of the target, action 4
      (no-op / full stop) is sent every tick until the target is moved
      (distance grows past threshold * 1.4) or the user quits.
    - When either detector fails, an error message is printed identifying
      which one failed, action 4 is sent, and detection is retried on the
      next frame.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "Target_Tracking"))
sys.path.insert(0, os.path.join(HERE, "Bluetooth_Comms"))

from car_detector import (  # noqa: E402
    DetectionSmoother,
    _configure_capture,
    _open_camera,
    detect_physical_car,
)
from state_generator import (  # noqa: E402
    DEBUG_WINDOW_NAME,
    MAP_HEIGHT,
    MAP_WIDTH,
    _assemble_vector,
    _draw_hud_text,
    _prepare_debug_window,
    _render_failure_view,
    _resize_to_map,
    render_debug_view,
)
from target_detector import detect_target  # noqa: E402
from remote_control import COMMANDS, PORT, send_command  # noqa: E402

from dqn_agent import DQNAgent  # noqa: E402


ACTION_LABELS = {
    0: "forward",
    1: "backward",
    2: "rotate_left",
    3: "rotate_right",
    4: "no_op",
}

NO_OP_ACTION = 4

TARGET_RADIUS_BUFFER = 1.20


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the trained DQN driving policy on the physical car using the "
            "overhead webcam for state estimation and a serial/Bluetooth link "
            "to the car for actuation."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["gui", "headless"],
        default="gui",
        help="'gui' opens an OpenCV debug window; 'headless' prints status only.",
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
        "--model",
        default="dqn_model.pth",
        help="Path to the trained DQN weights (default: dqn_model.pth).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0 / 15.0,
        help="Seconds between processed frames (default: 1/15).",
    )
    parser.add_argument(
        "--reach-threshold",
        type=float,
        default=45.0,
        help=(
            "Pixel distance (in the resized 900x600 map) at which the car is "
            "considered to have reached the target (default: 45.0)."
        ),
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help=(
            "Skip opening the serial port; actions are only printed. Useful "
            "for testing the perception + DQN path without the car attached."
        ),
    )
    return parser


def _open_serial(port: str, baud: int):
    import serial  # local import so --no-serial works without pyserial installed

    return serial.Serial(port, baud, timeout=1)


def _warm_up_camera(capture, num_frames: int = 5) -> None:
    """Read and discard a few frames so the webcam settles its exposure."""
    for _ in range(num_frames):
        ok, _frame = capture.read()
        if not ok:
            break


def _grab_frame(capture) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read a frame from the webcam.")
    return frame


def _pixel_distance(car: dict, tgt: dict) -> float:
    cx, cy = car["car_center"]
    tx, ty = tgt["center"]
    return math.hypot(tx - cx, ty - cy)


def _send_action(ser, action: int, no_serial: bool) -> None:
    cmd = str(action)
    if cmd not in COMMANDS:
        raise ValueError(f"Unknown action id: {action}")
    if no_serial or ser is None:
        # Mirror send_command's stdout format so logs look the same in both modes.
        print(f"Sent (mock): {COMMANDS[cmd].strip()}")
        return
    send_command(ser, cmd)


def _draw_hud(canvas: np.ndarray, lines: list[str], start_y: int = 110) -> None:
    """Draw a stack of HUD lines on top of an existing debug view."""
    y = start_y
    for line in lines:
        _draw_hud_text(canvas, line, (12, y))
        y += 22


def _check_quit_gui() -> bool:
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def _check_quit_headless() -> bool:
    if os.name != "nt":
        return False
    import msvcrt

    if not msvcrt.kbhit():
        return False
    try:
        ch = msvcrt.getwch()
    except Exception:
        return False
    return ch.lower() == "q"


def run(args: argparse.Namespace) -> None:
    if args.mode == "headless" and os.name != "nt":
        raise RuntimeError(
            "Headless quit-on-'q' is implemented via msvcrt and only supported "
            "on Windows. Use --mode gui on this platform."
        )

    print(f"[BirdsEyeRun] Loading DQN weights from {args.model} ...")
    agent = DQNAgent()
    agent.load(args.model)
    agent.epsilon = 0.0
    print(f"[BirdsEyeRun] DQN ready on device {agent.device}.")

    ser = None
    if not args.no_serial:
        print(f"[BirdsEyeRun] Opening serial port {args.port} @ {args.baud} ...")
        ser = _open_serial(args.port, args.baud)
        print("[BirdsEyeRun] Serial connected.")
    else:
        print("[BirdsEyeRun] --no-serial set; actions will be printed only.")

    print(f"[BirdsEyeRun] Opening camera index {args.camera_index} ...")
    capture = _open_camera(args.camera_index)
    _configure_capture(capture, 1920, 1080)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")
    _warm_up_camera(capture)
    print("[BirdsEyeRun] Camera ready.")

    if args.mode == "gui":
        _prepare_debug_window()
        print("[BirdsEyeRun] GUI mode active. Press 'q' in the debug window to quit.")
    else:
        print("[BirdsEyeRun] Headless mode active. Press 'q' in this terminal to quit.")

    smoother = DetectionSmoother()
    reached = False
    last_tick = 0.0

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - last_tick
            if elapsed < args.interval:
                # Stay responsive to the quit key while we wait for the next tick.
                if args.mode == "gui":
                    if _check_quit_gui():
                        break
                else:
                    if _check_quit_headless():
                        break
                    time.sleep(min(0.005, args.interval - elapsed))
                continue
            last_tick = now

            frame = _grab_frame(capture)
            resized = _resize_to_map(frame)

            car = None
            tgt = None
            failure_reason: str | None = None

            try:
                car, _ = detect_physical_car(resized)
            except ValueError as exc:
                failure_reason = f"[car] not detected: {exc}"

            if car is not None:
                try:
                    tgt = detect_target(resized)
                except ValueError as exc:
                    failure_reason = f"[target] not detected: {exc}"

            if car is None or tgt is None:
                # Detection failure: stop the car, keep the loop alive.
                print(failure_reason)
                _send_action(ser, NO_OP_ACTION, args.no_serial)
                reached = False

                if args.mode == "gui":
                    which = "car" if car is None else "target"
                    view = _render_failure_view(resized, f"{which} not detected")
                    _draw_hud(
                        view,
                        [
                            "state=FAILED",
                            f"action={NO_OP_ACTION} ({ACTION_LABELS[NO_OP_ACTION]})",
                        ],
                        start_y=90,
                    )
                    cv2.imshow(DEBUG_WINDOW_NAME, view)
                    if _check_quit_gui():
                        break
                else:
                    if _check_quit_headless():
                        break
                continue

            car = smoother.update(car)
            tgt["radius"] = float(tgt["radius"]) * TARGET_RADIUS_BUFFER

            vector = _assemble_vector(car, tgt)
            dist_px = _pixel_distance(car, tgt)

            if reached:
                # Hysteresis: stay 'reached' until the target moves clearly away.
                if dist_px > args.reach_threshold * 1.4:
                    print(
                        f"[BirdsEyeRun] Target moved (dist={dist_px:.1f} px). "
                        "Resuming driving."
                    )
                    reached = False
            else:
                if dist_px <= args.reach_threshold:
                    print(
                        f"[BirdsEyeRun] TARGET REACHED (dist={dist_px:.1f} px). "
                        "Holding position. Move the target or press 'q' to quit."
                    )
                    reached = True

            if reached:
                action = NO_OP_ACTION
                state_label = "REACHED"
            else:
                action = int(agent.select_action(vector, evaluate=True))
                state_label = "DRIVING"

            _send_action(ser, action, args.no_serial)

            if args.mode == "gui":
                view = render_debug_view(resized, car, tgt, vector)
                _draw_hud(
                    view,
                    [
                        f"state={state_label}",
                        f"action={action} ({ACTION_LABELS[action]})",
                        f"dist={dist_px:6.1f} px  thresh={args.reach_threshold:.1f}",
                    ],
                )
                cv2.imshow(DEBUG_WINDOW_NAME, view)
                if _check_quit_gui():
                    break
            else:
                print(
                    f"[t={now:9.3f}] action={action} ({ACTION_LABELS[action]:<13}) "
                    f"dist={dist_px:6.1f} px  state={state_label}"
                )
                if _check_quit_headless():
                    break

    finally:
        print("[BirdsEyeRun] Shutting down ...")
        try:
            _send_action(ser, NO_OP_ACTION, args.no_serial)
        except Exception as exc:
            print(f"[BirdsEyeRun] Warning: failed to send stop command: {exc}")

        if ser is not None:
            try:
                if ser.is_open:
                    ser.close()
                    print("[BirdsEyeRun] Serial disconnected.")
            except Exception as exc:
                print(f"[BirdsEyeRun] Warning: failed to close serial: {exc}")

        try:
            capture.release()
            print("[BirdsEyeRun] Camera released.")
        except Exception as exc:
            print(f"[BirdsEyeRun] Warning: failed to release camera: {exc}")

        if args.mode == "gui":
            cv2.destroyAllWindows()

        print("[BirdsEyeRun] Done.")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

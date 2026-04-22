"""
List cameras (index probe), print OpenCV properties, preview with FPS overlay.
Windows-focused: MSMF or DirectShow backends.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from typing import Iterable

import cv2

WINDOW_NAME = "Camera feed"
ROLLING_FRAMES = 30
# Request a high FPS; driver clamps to hardware maximum.
REQUEST_FPS = 240


def backend_from_name(name: str) -> int:
    n = name.lower().strip()
    if n == "msmf":
        return cv2.CAP_MSMF
    if n == "dshow":
        return cv2.CAP_DSHOW
    raise ValueError(f"Unknown backend: {name!r} (use msmf or dshow)")


def fourcc_to_str(code: float) -> str:
    i = int(code)
    if i == 0:
        return "n/a"
    return "".join(chr((i >> (8 * k)) & 0xFF) for k in range(4))


def probe_cameras(max_index: int, backend: int) -> list[int]:
    """Return indices in [0, max_index) that open and yield at least one frame."""
    found: list[int] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        try:
            if not cap.isOpened():
                continue
            ok, _ = cap.read()
            if ok:
                found.append(i)
        finally:
            cap.release()
    return found


def _try_set(cap: cv2.VideoCapture, prop_id: int, value: float, label: str) -> None:
    before = cap.get(prop_id)
    ret = cap.set(prop_id, value)
    after = cap.get(prop_id)
    print(f"  {label}: set({value}) returned {ret}; get before={before!r} after={after!r}")


def apply_capture_tuning(
    cap: cv2.VideoCapture,
    width: int | None,
    height: int | None,
    request_fps: float,
) -> None:
    """Apply settings that often improve throughput / reduce backlog."""
    print("Capture tuning (read-back shows what stuck):")
    _try_set(cap, cv2.CAP_PROP_BUFFERSIZE, 1, "BUFFERSIZE")
    if width is not None:
        _try_set(cap, cv2.CAP_PROP_FRAME_WIDTH, float(width), "FRAME_WIDTH")
    if height is not None:
        _try_set(cap, cv2.CAP_PROP_FRAME_HEIGHT, float(height), "FRAME_HEIGHT")
    _try_set(cap, cv2.CAP_PROP_FPS, float(request_fps), "FPS")
    print()


def camera_property_rows() -> Iterable[tuple[str, int]]:
    """Human-readable name and CAP_PROP id for metadata dump."""
    props = [
        ("FRAME_WIDTH", cv2.CAP_PROP_FRAME_WIDTH),
        ("FRAME_HEIGHT", cv2.CAP_PROP_FRAME_HEIGHT),
        ("FPS", cv2.CAP_PROP_FPS),
        ("FOURCC", cv2.CAP_PROP_FOURCC),
        ("BUFFERSIZE", cv2.CAP_PROP_BUFFERSIZE),
        ("BRIGHTNESS", cv2.CAP_PROP_BRIGHTNESS),
        ("CONTRAST", cv2.CAP_PROP_CONTRAST),
        ("SATURATION", cv2.CAP_PROP_SATURATION),
        ("HUE", cv2.CAP_PROP_HUE),
        ("GAIN", cv2.CAP_PROP_GAIN),
        ("EXPOSURE", cv2.CAP_PROP_EXPOSURE),
        ("AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE),
        ("AUTOFOCUS", cv2.CAP_PROP_AUTOFOCUS),
        ("FOCUS", cv2.CAP_PROP_FOCUS),
        ("ZOOM", cv2.CAP_PROP_ZOOM),
        ("IRIS", cv2.CAP_PROP_IRIS),
        ("TEMPERATURE", cv2.CAP_PROP_TEMPERATURE),
        ("BACKEND", cv2.CAP_PROP_BACKEND),
        ("CODEC_PIXEL_FORMAT", cv2.CAP_PROP_CODEC_PIXEL_FORMAT),
        ("FORMAT", cv2.CAP_PROP_FORMAT),
        ("MODE", cv2.CAP_PROP_MODE),
        ("POS_MSEC", cv2.CAP_PROP_POS_MSEC),
        ("POS_FRAMES", cv2.CAP_PROP_POS_FRAMES),
    ]
    return props


def print_camera_info(cap: cv2.VideoCapture) -> None:
    print("Camera information (OpenCV properties; 0 often means unsupported or unknown):")
    print("-" * 60)
    backend_name = cap.getBackendName()
    print(f"  backend_name: {backend_name}")
    for label, pid in camera_property_rows():
        v = cap.get(pid)
        if pid == cv2.CAP_PROP_FOURCC:
            print(f"  {label}: {v!r} ({fourcc_to_str(v)})")
        else:
            print(f"  {label}: {v!r}")
    print("-" * 60)
    print(
        "Note: Some drivers return 0 for optional controls even when streaming works.\n"
    )


def _window_still_open() -> bool:
    try:
        vis = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        return vis >= 1
    except cv2.error:
        return False


def run_preview(cap: cv2.VideoCapture) -> None:
    times: collections.deque[float] = collections.deque(maxlen=ROLLING_FRAMES + 1)
    times.append(time.perf_counter())

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("read() failed; exiting preview.")
            break

        now = time.perf_counter()
        times.append(now)
        if len(times) >= 2:
            span = times[-1] - times[0]
            fps = (len(times) - 1) / span if span > 0 else 0.0
        else:
            fps = 0.0

        label = f"FPS: {fps:.1f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x, y = 10, 30
        cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 6), (0, 0, 0), -1)
        cv2.putText(
            frame,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if not _window_still_open():
            break

    cv2.destroyWindow(WINDOW_NAME)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List cameras, show metadata, preview with FPS.")
    p.add_argument(
        "--max-cameras",
        type=int,
        default=10,
        metavar="N",
        help="Probe indices 0 .. N-1 (default: 10)",
    )
    p.add_argument(
        "--backend",
        choices=("msmf", "dshow"),
        default="msmf",
        help="Video capture backend (default: msmf)",
    )
    p.add_argument("--width", type=int, default=None, help="Request frame width (optional)")
    p.add_argument("--height", type=int, default=None, help="Request frame height (optional)")
    p.add_argument(
        "--request-fps",
        type=float,
        default=REQUEST_FPS,
        metavar="FPS",
        help=f"Request max FPS from driver (default: {REQUEST_FPS}; actual rate is clamped)",
    )
    return p.parse_args(argv)


def prompt_choice(indices: list[int]) -> int:
    while True:
        raw = input(f"Select camera [0-{len(indices) - 1}] (or q to quit): ").strip().lower()
        if raw == "q":
            raise SystemExit(0)
        if not raw.isdigit():
            print("Enter a non-negative integer index from the list.")
            continue
        i = int(raw)
        if 0 <= i < len(indices):
            return indices[i]
        print(f"Pick a number between 0 and {len(indices) - 1}.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backend = backend_from_name(args.backend)

    print(f"Probing cameras (backend={args.backend}, indices 0..{args.max_cameras - 1})...")
    found = probe_cameras(args.max_cameras, backend)
    if not found:
        print("No cameras found. Try --backend dshow or increase --max-cameras.")
        return 1

    print("Available camera indices:")
    for j, idx in enumerate(found):
        print(f"  [{j}] -> device index {idx}")

    chosen_index = prompt_choice(found)

    cap = cv2.VideoCapture(chosen_index, backend)
    if not cap.isOpened():
        print(f"Could not open camera index {chosen_index}.")
        return 1

    try:
        apply_capture_tuning(cap, args.width, args.height, args.request_fps)
        print_camera_info(cap)
        print("Preview: press 'q' or close the window to quit.\n")
        run_preview(cap)
    finally:
        cap.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())

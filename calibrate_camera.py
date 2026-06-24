"""Camera intrinsics calibration using a checkerboard.

Captures frames from the D435i (or any OpenCV-compatible camera), detects
checkerboard corners, and runs cv2.calibrateCamera to compute the camera
matrix and distortion coefficients.  Results are written back to config.yaml.

Usage:
    python calibrate_camera.py --config config.yaml

Controls during capture:
    SPACE  — capture current frame (must detect corners)
    Q      — quit / cancel
    C      — run calibration now (when enough frames collected)

Recommended: collect 20-30 frames from varied angles and distances.

Checkerboard: print an asymmetric checkerboard (inner corners, not squares).
Default: 9x6 inner corners (10x7 squares).  Pass --cols and --rows to change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

_MIN_FRAMES = 10


def _open_camera(camera_index: int):
    """Return (get_frame_fn, cleanup_fn) for RealSense or OpenCV fallback."""
    if _RS_AVAILABLE:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(cfg)

        def get_frame():
            frames = pipeline.wait_for_frames()
            return np.asanyarray(frames.get_color_frame().get_data())

        return get_frame, pipeline.stop
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

        def get_frame():
            ok, frame = cap.read()
            return frame if ok else None

        return get_frame, cap.release


def calibrate(config_path: str, cols: int, rows: int, square_m: float,
              camera_index: int) -> None:
    board = (cols, rows)
    objp = np.zeros((cols * rows, 3), dtype=float)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    obj_points: list = []
    img_points: list = []
    image_size = None

    get_frame, cleanup = _open_camera(camera_index)
    print(f"Checkerboard: {cols}x{rows} inner corners, {square_m*1000:.0f} mm squares")
    print("SPACE=capture  C=calibrate  Q=quit")

    try:
        while True:
            frame = get_frame()
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]

            found, corners = cv2.findChessboardCorners(gray, board, None)
            display = frame.copy()
            if found:
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                cv2.drawChessboardCorners(display, board, corners2, found)

            n = len(obj_points)
            status = f"Captured: {n}  (need {_MIN_FRAMES}+)  {'CORNERS FOUND' if found else 'no corners'}"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 200, 0) if found else (0, 0, 200), 2)
            cv2.imshow("Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Cancelled.")
                return
            elif key == ord(" ") and found:
                obj_points.append(objp)
                img_points.append(corners2)
                print(f"  Frame {len(obj_points)} captured.")
            elif key == ord("c"):
                if len(obj_points) < _MIN_FRAMES:
                    print(f"Need at least {_MIN_FRAMES} frames, have {len(obj_points)}.")
                else:
                    break
    finally:
        cleanup()
        cv2.destroyAllWindows()

    if len(obj_points) < _MIN_FRAMES:
        print("Not enough frames — calibration aborted.")
        sys.exit(1)

    print(f"\nCalibrating with {len(obj_points)} frames...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)
    print(f"RMS reprojection error: {rms:.4f} px")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Dist coeffs: {dist_coeffs.flatten()}")

    # Write to config.yaml
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "hardware" not in raw or raw["hardware"] is None:
        raw["hardware"] = {}
    raw["hardware"]["camera_matrix"] = camera_matrix.flatten().tolist()
    raw["hardware"]["dist_coeffs"] = dist_coeffs.flatten().tolist()
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved to {config_path}")
    print("Next step: run calibrate_handeye.py (CAL-02)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--cols", type=int, default=9, help="Inner corner columns (default 9)")
    p.add_argument("--rows", type=int, default=6, help="Inner corner rows (default 6)")
    p.add_argument("--square", type=float, default=0.025, help="Square size in metres (default 0.025)")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index (ignored if pyrealsense2 present)")
    args = p.parse_args()
    calibrate(args.config, args.cols, args.rows, args.square, args.camera)

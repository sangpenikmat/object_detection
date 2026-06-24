"""ArUco marker pose estimator for the D435i eye-in-hand camera.

Detects a single ArUco marker in the D435i RGB stream and returns the
marker's position in robot-base coordinates using the hand-eye transform
T_cam_to_base stored in config.yaml.

Standalone test:
    python -m ur5_tracking.aruco_detector --config config.yaml

Output when used as a library:
    detector = ArucoDetector(cfg)
    detector.start()
    pos = detector.get_object_pos()   # np.ndarray shape (3,), robot-base frame
    speed = detector.get_object_speed()
    detector.stop()
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False


class ArucoDetector:
    """Thread-safe ArUco pose estimator feeding HardwareInterface."""

    def __init__(self, cfg) -> None:
        hw = cfg.get("hardware") or {}

        # ArUco setup
        dict_name = hw.get("aruco_dict", "DICT_4X4_50")
        aruco_dict_id = getattr(cv2.aruco, dict_name)
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        self._marker_id = int(hw.get("marker_id", 0))
        self._marker_size = float(hw.get("marker_size_m", 0.05))

        # Camera intrinsics
        cam_mat_flat = hw.get("camera_matrix", [1, 0, 0, 0, 1, 0, 0, 0, 1])
        self._camera_matrix = np.array(cam_mat_flat, dtype=float).reshape(3, 3)
        dist_flat = hw.get("dist_coeffs", [0, 0, 0, 0, 0])
        self._dist_coeffs = np.array(dist_flat, dtype=float)

        # Hand-eye transform: camera frame → robot base frame (row-major 4x4)
        T_flat = hw.get("T_cam_to_base", [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
        self._T_cam_to_base = np.array(T_flat, dtype=float).reshape(4, 4)

        # 3D marker corners in marker frame (z=0 plane, centred)
        h = self._marker_size / 2.0
        self._obj_points = np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0],
        ], dtype=float)

        # RealSense pipeline
        self._camera_index = int(hw.get("camera_index", 0))
        self._pipeline: Optional[object] = None  # rs.pipeline when running

        # State protected by lock
        self._lock = threading.Lock()
        self._pos: Optional[np.ndarray] = None
        self._pos_prev: Optional[np.ndarray] = None
        self._t_prev: float = 0.0
        self._speed: float = 0.0
        self._last_seen: float = 0.0

        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not _RS_AVAILABLE:
            raise RuntimeError(
                "pyrealsense2 not installed. Run: pip install pyrealsense2"
            )
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._pipeline.start(config)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._pipeline:
            self._pipeline.stop()

    # ------------------------------------------------------------------
    def get_object_pos(self) -> Optional[np.ndarray]:
        """Return latest marker position in robot-base frame, or None if not seen."""
        with self._lock:
            return self._pos.copy() if self._pos is not None else None

    def get_object_speed(self) -> float:
        """Return latest estimated marker speed (m/s) in base frame."""
        with self._lock:
            return self._speed

    def last_seen_age(self) -> float:
        """Seconds since the marker was last successfully detected."""
        with self._lock:
            return time.monotonic() - self._last_seen if self._last_seen > 0 else float("inf")

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=200)
            except Exception:
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            pos_cam = self._detect(frame)
            if pos_cam is not None:
                self._update(pos_cam)

    def _detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect the target marker and return its position in camera frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return None
        for i, mid in enumerate(ids.flatten()):
            if mid != self._marker_id:
                continue
            ok, rvec, tvec = cv2.solvePnP(
                self._obj_points,
                corners[i].reshape(4, 2),
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                return tvec.flatten()  # (x,y,z) in camera frame, metres
        return None

    def _update(self, pos_cam: np.ndarray) -> None:
        pos_h = np.append(pos_cam, 1.0)
        pos_base = (self._T_cam_to_base @ pos_h)[:3]
        now = time.monotonic()
        with self._lock:
            if self._pos_prev is not None and self._t_prev > 0:
                dt = now - self._t_prev
                if dt > 0:
                    self._speed = float(np.linalg.norm(pos_base - self._pos_prev) / dt)
            self._pos_prev = self._pos.copy() if self._pos is not None else pos_base.copy()
            self._t_prev = now
            self._pos = pos_base
            self._last_seen = now


# ---------------------------------------------------------------------------
# Standalone test: display live detections in an OpenCV window
# ---------------------------------------------------------------------------
def _run_test(cfg_path: str) -> None:
    from .config_loader import load_config
    cfg = load_config(cfg_path)
    hw = cfg.get("hardware") or {}

    dict_name = hw.get("aruco_dict", "DICT_4X4_50")
    aruco_dict_id = getattr(cv2.aruco, dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    marker_id = int(hw.get("marker_id", 0))
    marker_size = float(hw.get("marker_size_m", 0.05))
    cam_mat = np.array(hw.get("camera_matrix", [1,0,0,0,1,0,0,0,1]), dtype=float).reshape(3,3)
    dist = np.array(hw.get("dist_coeffs", [0,0,0,0,0]), dtype=float)
    T = np.array(hw.get("T_cam_to_base", [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]), dtype=float).reshape(4,4)
    h = marker_size / 2.0
    obj_pts = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], dtype=float)

    if _RS_AVAILABLE:
        pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(rs_cfg)
        def get_frame():
            frames = pipeline.wait_for_frames()
            return np.asanyarray(frames.get_color_frame().get_data())
        cleanup = pipeline.stop
    else:
        cap = cv2.VideoCapture(int(hw.get("camera_index", 0)))
        def get_frame():
            _, frame = cap.read()
            return frame
        cleanup = cap.release

    print("Press Q to quit.")
    try:
        while True:
            frame = get_frame()
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    if mid != marker_id:
                        continue
                    ok, rvec, tvec = cv2.solvePnP(
                        obj_pts, corners[i].reshape(4,2), cam_mat, dist,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        cv2.drawFrameAxes(frame, cam_mat, dist, rvec, tvec, marker_size*0.5)
                        pos_base = (T @ np.append(tvec.flatten(), 1.0))[:3]
                        label = f"base XYZ: {pos_base[0]:.3f} {pos_base[1]:.3f} {pos_base[2]:.3f} m"
                        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)
            cv2.imshow("ArUco detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    _run_test(args.config)

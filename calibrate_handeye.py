"""Hand-eye calibration: compute T_cam_to_base (camera → robot base).

Moves the UR5e arm to N poses, records the end-effector pose from RTDE and
the ArUco marker pose from the D435i at each configuration, then solves for
the camera-to-end-effector transform using cv2.calibrateHandEye.

The result (T_cam_to_base) is saved to config.yaml.

Usage:
    python calibrate_handeye.py --config config.yaml --poses 15

Requirements (Phase 2 hardware):
    pip install ur-rtde pyrealsense2 opencv-contrib-python

Procedure:
    1. Mount the calibration marker RIGIDLY somewhere fixed in the scene
       (NOT on the robot or gripper — it must be stationary).
    2. Run this script. The arm moves to each pose automatically.
    3. Confirm each capture looks good (marker visible, no blur).
    4. The transform is saved to config.yaml hardware.T_cam_to_base.

The poses are defined as joint-space waypoints spread around the workspace
so the arm views the marker from varied angles — edit POSE_SET below if
the default positions are outside your specific workspace.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Joint-space waypoints (radians) for 15-pose calibration grid.
# These assume the marker is placed in front of the robot at ~0.4 m height.
# Adjust if your workspace differs.
# ---------------------------------------------------------------------------
POSE_SET = [
    [-1.40, -1.30,  1.50, -1.75, -1.57,  0.00],
    [-1.40, -1.30,  1.50, -1.75, -1.57,  0.50],
    [-1.40, -1.30,  1.50, -1.75, -1.57, -0.50],
    [-1.00, -1.30,  1.50, -1.75, -1.57,  0.00],
    [-1.80, -1.30,  1.50, -1.75, -1.57,  0.00],
    [-1.40, -1.10,  1.30, -1.75, -1.57,  0.00],
    [-1.40, -1.50,  1.70, -1.75, -1.57,  0.00],
    [-1.40, -1.30,  1.50, -1.55, -1.57,  0.00],
    [-1.40, -1.30,  1.50, -1.95, -1.57,  0.00],
    [-1.00, -1.10,  1.30, -1.55, -1.57,  0.30],
    [-1.80, -1.10,  1.30, -1.55, -1.57, -0.30],
    [-1.00, -1.50,  1.70, -1.95, -1.57,  0.30],
    [-1.80, -1.50,  1.70, -1.95, -1.57, -0.30],
    [-1.40, -1.30,  1.50, -1.75, -1.20,  0.00],
    [-1.40, -1.30,  1.50, -1.75, -1.90,  0.00],
]


def _rvec_tvec_from_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.flatten(), T[:3, 3]


def _pose_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = tvec
    return T


def _detect_marker(cap_fn, aruco_dict, detector_params, marker_id,
                   marker_size, camera_matrix, dist_coeffs):
    """Grab a frame and return (rvec, tvec) of the marker, or None."""
    h = marker_size / 2.0
    obj_pts = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], dtype=float)
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    for _ in range(10):  # retry up to 10 frames
        frame = cap_fn()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        for i, mid in enumerate(ids.flatten()):
            if mid != marker_id:
                continue
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, corners[i].reshape(4, 2),
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                return rvec.flatten(), tvec.flatten()
    return None


def calibrate(config_path: str, n_poses: int, arm_speed: float,
              arm_accel: float) -> None:
    cfg_file = Path(config_path)
    with cfg_file.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    hw = raw.get("hardware") or {}

    arm_ip = hw.get("arm_ip", "192.168.1.100")
    marker_id = int(hw.get("marker_id", 0))
    marker_size = float(hw.get("marker_size_m", 0.05))
    cam_mat = np.array(hw.get("camera_matrix", [1,0,0,0,1,0,0,0,1]), dtype=float).reshape(3,3)
    dist = np.array(hw.get("dist_coeffs", [0,0,0,0,0]), dtype=float)

    # Check calibration data looks real
    if np.allclose(cam_mat, np.eye(3)):
        print("WARNING: camera_matrix is identity — run calibrate_camera.py first.")

    poses = POSE_SET[:n_poses]
    if len(poses) < n_poses:
        print(f"Only {len(poses)} poses defined in POSE_SET; using all.")

    # --- connect to arm -------------------------------------------------------
    try:
        import rtde_control
        import rtde_receive
    except ImportError:
        print("ur-rtde not installed. Run: pip install ur-rtde")
        sys.exit(1)

    print(f"Connecting to UR5e at {arm_ip}...")
    ctrl = rtde_control.RTDEControlInterface(arm_ip)
    recv = rtde_receive.RTDEReceiveInterface(arm_ip)
    print("Connected.")

    # --- connect to camera ----------------------------------------------------
    try:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(rs_cfg)
        def get_frame():
            frames = pipeline.wait_for_frames()
            return np.asanyarray(frames.get_color_frame().get_data())
        cam_cleanup = pipeline.stop
    except ImportError:
        cam_idx = int(hw.get("camera_index", 0))
        cap = cv2.VideoCapture(cam_idx)
        def get_frame():
            ok, f = cap.read()
            return f if ok else None
        cam_cleanup = cap.release

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, hw.get("aruco_dict", "DICT_4X4_50")))
    aruco_params = cv2.aruco.DetectorParameters()

    R_gripper2base: List[np.ndarray] = []
    t_gripper2base: List[np.ndarray] = []
    R_target2cam:   List[np.ndarray] = []
    t_target2cam:   List[np.ndarray] = []

    try:
        for idx, q in enumerate(poses):
            print(f"\nPose {idx+1}/{len(poses)}: moving arm...")
            ctrl.moveJ(q, arm_speed, arm_accel)
            time.sleep(0.5)  # settle

            # End-effector pose in base frame (4x4)
            tcp = recv.getActualTCPPose()  # [x,y,z, rx,ry,rz]
            T_ee = np.eye(4)
            R_ee, _ = cv2.Rodrigues(np.array(tcp[3:]))
            T_ee[:3, :3] = R_ee
            T_ee[:3,  3] = tcp[:3]

            # Marker detection
            result = _detect_marker(get_frame, aruco_dict, aruco_params,
                                    marker_id, marker_size, cam_mat, dist)
            if result is None:
                print(f"  Marker not detected at pose {idx+1} — skipping.")
                continue

            rvec_m, tvec_m = result
            R_m, _ = cv2.Rodrigues(rvec_m)
            print(f"  Marker at cam-frame: {tvec_m}")

            R_gripper2base.append(T_ee[:3, :3])
            t_gripper2base.append(T_ee[:3,  3])
            R_target2cam.append(R_m)
            t_target2cam.append(tvec_m)
            print(f"  Recorded. ({len(R_gripper2base)} good poses)")

    finally:
        ctrl.stopScript()
        cam_cleanup()

    if len(R_gripper2base) < 4:
        print(f"Only {len(R_gripper2base)} usable poses — need at least 4. Aborting.")
        sys.exit(1)

    print(f"\nSolving hand-eye with {len(R_gripper2base)} poses (Tsai method)...")
    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,   t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI)

    # T_cam_to_base = T_ee_to_base @ T_cam_to_ee
    # We store the full 4x4 in row-major order.
    T_cam2ee = np.eye(4)
    T_cam2ee[:3, :3] = R_cam2ee
    T_cam2ee[:3,  3] = t_cam2ee.flatten()
    # Average base-to-EE over collected poses for a representative transform
    T_ee2base = np.eye(4)
    T_ee2base[:3, :3] = np.mean(R_gripper2base, axis=0)
    T_ee2base[:3,  3] = np.mean(t_gripper2base, axis=0)
    T_cam2base = T_ee2base @ T_cam2ee

    print("T_cam_to_base (cam → robot base):")
    print(np.round(T_cam2base, 5))

    # Save to config.yaml
    if "hardware" not in raw or raw["hardware"] is None:
        raw["hardware"] = {}
    raw["hardware"]["T_cam_to_base"] = T_cam2base.flatten().tolist()
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved T_cam_to_base to {config_path}")
    print("Hand-eye calibration complete. You can now run in hardware mode.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--poses", type=int, default=15, help="Number of calibration poses (default 15)")
    p.add_argument("--speed", type=float, default=0.3, help="moveJ speed rad/s (default 0.3)")
    p.add_argument("--accel", type=float, default=0.2, help="moveJ accel rad/s^2 (default 0.2)")
    args = p.parse_args()
    calibrate(args.config, args.poses, args.speed, args.accel)

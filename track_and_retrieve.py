#!/usr/bin/env python3
"""UR5e object tracking + retrieval — simulation and hardware.

The arm gaze-tracks a movable object (the gripper-mounted camera follows it).
When the object is left stationary, the arm picks it up and returns it to its
original position.

Simulation modes:
  python track_and_retrieve.py                              # interactive (drag the object)
  python track_and_retrieve.py --auto                       # object moves automatically
  python track_and_retrieve.py --headless --seconds 30      # headless verification
  python track_and_retrieve.py --headless --seconds 30 --record out.mp4

Hardware mode (Phase 2 — requires ur-rtde, pyrobotiqgripper, pyrealsense2):
  python track_and_retrieve.py --hardware
"""
from __future__ import annotations

import argparse
import sys
import time

import mujoco
import numpy as np

from ur5_tracking.config_loader import load_config
from ur5_tracking.object_scene import build_model
from ur5_tracking.robot_interface import RobotInterface
from ur5_tracking.sim_interface import SimInterface
from ur5_tracking.hardware_interface import HardwareInterface
from ur5_tracking.track_control import TrackController
from ur5_tracking.track_states import RetrieveFSM
from ur5_tracking.auto_object import AutoObjectDriver

CONTROL_EVERY = 5   # run the controller every N physics steps


def setup_sim(cfg):
    model, _ = build_model(cfg)
    data = mujoco.MjData(model)
    iface = SimInterface(model, data, cfg)
    ctl = TrackController(model, cfg, iface)
    perch = cfg.arr("tracking", "perch_joints")
    data.qpos[iface.qadr] = perch
    data.ctrl[iface.arm_act] = perch
    iface.set_object_pose(cfg.arr("object", "spawn"))
    mujoco.mj_forward(model, data)
    return model, data, ctl, RetrieveFSM(ctl, cfg)


def run_headless(cfg, seconds, record=None):
    model, data, ctl, fsm = setup_sim(cfg)
    driver = AutoObjectDriver(ctl, cfg)
    dt = model.opt.timestep
    renderer = mujoco.Renderer(model, height=480, width=640) if record else None
    frames, transitions, placements, last_state = [], [], [], None

    for i in range(int(seconds / dt)):
        if i % CONTROL_EVERY == 0:
            fsm.update()
            driver.update(fsm.state, ctl.get_time(), dt * CONTROL_EVERY)
            if fsm.state != last_state:
                transitions.append((round(ctl.get_time(), 2), fsm.state))
                # capture placement accuracy at the moment of release
                if fsm.state == "RETREAT":
                    placements.append(ctl.object_pos().copy())
                last_state = fsm.state
        mujoco.mj_step(model, data)
        if renderer and i % 20 == 0:
            renderer.update_scene(data, camera="gripper_cam")
            frames.append(renderer.render().copy())

    obj = ctl.object_pos()
    home = cfg.arr("home_return", "position")
    cycles = sum(1 for _, st in transitions if st == "GRASP")
    print("State transitions:", transitions)
    print(f"Completed retrieve cycles: {cycles}")
    if placements:
        errs = [float(np.linalg.norm(p[:2] - home[:2])) for p in placements]
        print("Placement xy errors (m):", [round(e, 3) for e in errs])
        print(f"Mean placement error: {np.mean(errs):.3f} m")
    else:
        print("No placement completed in the given time.")
    if record and frames:
        try:
            import imageio
            imageio.mimsave(record, frames, fps=30)
            print("Saved video:", record)
        except ImportError:
            print("(imageio not installed; skipping video)")


def run_viewer(cfg, auto):
    import mujoco.viewer
    model, data, ctl, fsm = setup_sim(cfg)
    driver = AutoObjectDriver(ctl, cfg) if auto else None
    dt = model.opt.timestep
    with mujoco.viewer.launch_passive(model, data) as v:
        i, last = 0, None
        while v.is_running():
            t0 = time.time()
            if i % CONTROL_EVERY == 0:
                fsm.update()
                if driver:
                    driver.update(fsm.state, ctl.get_time(), dt * CONTROL_EVERY)
                if fsm.state != last:
                    print(f"[{ctl.get_time():6.2f}s] -> {fsm.state}")
                    last = fsm.state
            mujoco.mj_step(model, data)
            v.sync()
            i += 1
            lag = dt - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)


def run_hardware(cfg) -> None:
    """Hardware control loop — connects to real UR5e, gripper, and D435i."""
    model, _ = build_model(cfg)
    iface = HardwareInterface(model, cfg)
    iface.connect()
    ctl = TrackController(model, cfg, iface)
    fsm = RetrieveFSM(ctl, cfg)
    dt = 1.0 / float(cfg.get("hardware", "control_hz", default=10))
    last_state = None
    print("[HW] Starting hardware control loop. Ctrl-C to stop.")
    try:
        while True:
            t0 = time.monotonic()
            fsm.update()
            if fsm.state != last_state:
                print(f"[{iface.get_time():.2f}s] -> {fsm.state}")
                last_state = fsm.state
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        print("\n[HW] Interrupted.")
    finally:
        iface.disconnect()


def main():
    ap = argparse.ArgumentParser(description="UR5e object tracking + retrieval")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--hardware", action="store_true",
                    help="run on real hardware (requires ur-rtde, pyrobotiqgripper, pyrealsense2)")
    ap.add_argument("--auto", action="store_true", help="(sim) move the object automatically")
    ap.add_argument("--headless", action="store_true", help="(sim) run without a display")
    ap.add_argument("--seconds", type=float, default=25.0, help="(sim) headless run duration")
    ap.add_argument("--record", default=None, help="(sim) save gripper-camera video (mp4)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.hardware:
        run_hardware(cfg)
    elif args.headless:
        run_headless(cfg, args.seconds, args.record)
    else:
        run_viewer(cfg, auto=args.auto)
    return 0


if __name__ == "__main__":
    sys.exit(main())

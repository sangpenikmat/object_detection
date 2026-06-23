#!/usr/bin/env python3
"""UR5e object tracking + retrieval simulation (MuJoCo, closed-loop).

The arm gaze-tracks a movable object (the gripper-mounted camera follows it).
When the object is left stationary, the arm picks it up and returns it to its
original position.

This is a SIMULATION: every step reads the object pose from the simulator's
ground truth. It is not an open-loop trajectory for a real robot.

Modes:
  python track_and_retrieve.py             # interactive: drag the object (Ctrl + right-drag)
  python track_and_retrieve.py --auto      # the object moves automatically, then is left
  python track_and_retrieve.py --headless --seconds 30          # no display (verification)
  python track_and_retrieve.py --headless --seconds 30 --record out.mp4   # record gripper cam
"""
from __future__ import annotations

import argparse
import sys
import time

import mujoco
import numpy as np

from ur5_tracking.config_loader import load_config
from ur5_tracking.object_scene import build_model
from ur5_tracking.track_control import TrackController
from ur5_tracking.track_states import RetrieveFSM
from ur5_tracking.auto_object import AutoObjectDriver

CONTROL_EVERY = 5   # run the controller every N physics steps


def setup(cfg):
    model, _ = build_model(cfg)
    data = mujoco.MjData(model)
    ctl = TrackController(model, cfg)
    perch = cfg.arr("tracking", "perch_joints")
    data.qpos[ctl.qadr] = perch
    data.ctrl[ctl.arm_act] = perch
    ctl.set_object_pose(data, cfg.arr("object", "spawn"))
    mujoco.mj_forward(model, data)
    return model, data, ctl, RetrieveFSM(ctl, cfg)


def run_headless(cfg, seconds, record=None):
    model, data, ctl, fsm = setup(cfg)
    driver = AutoObjectDriver(ctl, cfg)
    dt = model.opt.timestep
    renderer = mujoco.Renderer(model, height=480, width=640) if record else None
    frames, transitions, placements, last_state = [], [], [], None

    for i in range(int(seconds / dt)):
        if i % CONTROL_EVERY == 0:
            fsm.update(data)
            driver.update(data, fsm.state, data.time, dt * CONTROL_EVERY)
            if fsm.state != last_state:
                transitions.append((round(data.time, 2), fsm.state))
                # capture placement accuracy at the moment of release
                if fsm.state == "RETREAT":
                    placements.append(ctl.object_pos(data).copy())
                last_state = fsm.state
        mujoco.mj_step(model, data)
        if renderer and i % 20 == 0:
            renderer.update_scene(data, camera="gripper_cam")
            frames.append(renderer.render().copy())

    obj = ctl.object_pos(data)
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
    model, data, ctl, fsm = setup(cfg)
    driver = AutoObjectDriver(ctl, cfg) if auto else None
    dt = model.opt.timestep
    with mujoco.viewer.launch_passive(model, data) as v:
        i, last = 0, None
        while v.is_running():
            t0 = time.time()
            if i % CONTROL_EVERY == 0:
                fsm.update(data)
                if driver:
                    driver.update(data, fsm.state, data.time, dt * CONTROL_EVERY)
                if fsm.state != last:
                    print(f"[{data.time:6.2f}s] -> {fsm.state}")
                    last = fsm.state
            mujoco.mj_step(model, data)
            v.sync()
            i += 1
            lag = dt - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)


def main():
    ap = argparse.ArgumentParser(description="UR5e object tracking + retrieval (MuJoCo)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--auto", action="store_true", help="move the object automatically")
    ap.add_argument("--headless", action="store_true", help="run without a display (verification)")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--record", default=None, help="save gripper-camera video (mp4)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.headless:
        run_headless(cfg, args.seconds, args.record)
    else:
        run_viewer(cfg, auto=args.auto)
    return 0


if __name__ == "__main__":
    sys.exit(main())

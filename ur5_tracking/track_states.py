"""State machine: gaze-track -> detect the object is left -> pick -> return.

update(d) is called every control tick inside the physics loop (non-blocking)
and writes d.ctrl. Phases:

  TRACK    : hold the perch pose, aim the camera/tool +z at the object.
             If the object stays slower than idle_speed for idle_time AND is not
             already at home -> start retrieving it.
  APPROACH : move to a hover point above the object (gripper open)
  DESCEND  : lower onto the object
  GRASP    : close the gripper and enable the weld
  LIFT     : raise back to the hover point
  CARRY    : move to the hover point above home
  PLACE    : lower onto home
  RELEASE  : open the gripper and disable the weld
  RETREAT  : raise again, then go back to TRACK

Every motion phase has an 8 s safety timeout so it cannot deadlock.
"""
from __future__ import annotations

import mujoco
import numpy as np

TRACK, APPROACH, DESCEND, GRASP, LIFT, CARRY, PLACE, RELEASE, RETREAT = (
    "TRACK", "APPROACH", "DESCEND", "GRASP", "LIFT", "CARRY", "PLACE", "RELEASE", "RETREAT")

_DOWN = np.array([0.0, 0.0, -1.0])


class RetrieveFSM:
    def __init__(self, ctl, cfg):
        self.ctl = ctl
        self.state = TRACK

        self.perch = cfg.arr("tracking", "perch_joints")
        self.idle_speed = float(cfg.get("tracking", "idle_speed", default=0.02))
        self.idle_time = float(cfg.get("tracking", "idle_time", default=1.5))
        self.gaze_pw = float(cfg.get("tracking", "gaze_pos_weight", default=2.0))

        self.hover = float(cfg.get("pick", "hover_height", default=0.14))
        self.tol = float(cfg.get("pick", "reach_tol", default=0.015))
        self.grip_settle = float(cfg.get("pick", "gripper_settle_s", default=0.6))
        self.coarse_tol = max(self.tol * 3.0, 0.03)  # 'close enough' if motion stalls
        self.stall_time = 0.8                        # s without progress => advance
        self.home = cfg.arr("home_return", "position")
        self.home_thresh = 0.04  # object counts as "already home" within this radius

        # pinch position at the perch pose (gaze keeps the arm here)
        sd = ctl.scratch
        sd.qpos[:] = 0.0
        sd.qpos[ctl.qadr] = self.perch
        mujoco.mj_kinematics(ctl.m, sd)
        self.perch_pinch = sd.site_xpos[ctl.pinch].copy()

        self._idle_t0 = None
        self._phase_t0 = 0.0
        self._pick_pos = None
        self._best_dist = np.inf
        self._improve_t = 0.0

    # -- helpers --------------------------------------------------------------
    def _reached(self, d, target):
        return np.linalg.norm(self.ctl.pinch_pos(d) - target) < self.tol

    def _enter(self, d, state):
        self.state = state
        self._phase_t0 = d.time
        self._best_dist = np.inf
        self._improve_t = d.time

    def _timeout(self, d, limit=8.0):
        return (d.time - self._phase_t0) > limit

    # -- main dispatch --------------------------------------------------------
    def update(self, d):
        st = self.state
        if st == TRACK:
            return self._track(d)
        if st == APPROACH:
            return self._goto(d, self._pick_pos + [0, 0, self.hover], False, DESCEND)
        if st == DESCEND:
            return self._goto(d, self._pick_pos.copy(), False, GRASP)
        if st == GRASP:
            return self._grip(d, close=True, weld=True, after=LIFT)
        if st == LIFT:
            return self._goto(d, self._pick_pos + [0, 0, self.hover], True, CARRY)
        if st == CARRY:
            return self._goto(d, self.home + [0, 0, self.hover], True, PLACE)
        if st == PLACE:
            return self._goto(d, self.home + [0, 0, 0.006], True, RELEASE)
        if st == RELEASE:
            return self._grip(d, close=False, weld=False, after=RETREAT)
        if st == RETREAT:
            return self._goto(d, self.home + [0, 0, self.hover], False, TRACK,
                              on_done=self._reset_track)

    # -- phases ---------------------------------------------------------------
    def _track(self, d):
        ctl = self.ctl
        obj = ctl.object_pos(d)
        aim = obj - self.perch_pinch   # point the tool +z at the object
        q_des = ctl.solve_ik(self.perch_pinch, aim, ctl.arm_q(d),
                             pos_weight=self.gaze_pw, ori_weight=1.0, iters=60)
        ctl.command_arm(d, q_des)
        ctl.command_gripper(d, close=False)

        at_home = np.linalg.norm(obj[:2] - self.home[:2]) < self.home_thresh
        if ctl.object_speed(d) < self.idle_speed and not at_home:
            if self._idle_t0 is None:
                self._idle_t0 = d.time
            elif (d.time - self._idle_t0) >= self.idle_time:
                self._pick_pos = obj.copy()
                self._idle_t0 = None
                self._enter(d, APPROACH)
        else:
            self._idle_t0 = None
        return self.state

    def _goto(self, d, target, closed, after, on_done=None):
        target = np.asarray(target, dtype=float)
        ctl = self.ctl
        q_des = ctl.solve_ik(target, _DOWN, ctl.arm_q(d), pos_weight=1.0, ori_weight=0.5, iters=100)
        ctl.command_arm(d, q_des)
        ctl.command_gripper(d, close=closed)

        dist = float(np.linalg.norm(ctl.pinch_pos(d) - target))
        if dist < self._best_dist - 1e-4:
            self._best_dist, self._improve_t = dist, d.time
        stalled = (dist < self.coarse_tol) and (d.time - self._improve_t > self.stall_time)
        if dist < self.tol or stalled or self._timeout(d):
            if on_done:
                on_done()
            self._enter(d, after)
        return self.state

    def _grip(self, d, close, weld, after):
        self.ctl.command_gripper(d, close=close)
        if (d.time - self._phase_t0) >= self.grip_settle:
            self.ctl.set_grasp(d, on=weld)
            self._enter(d, after)
        return self.state

    def _reset_track(self):
        self.state = TRACK
        self._idle_t0 = None

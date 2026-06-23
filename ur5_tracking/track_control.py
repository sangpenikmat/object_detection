"""Low-level control utilities for the track / pick / return task.

TrackController reads sim state (gripper pinch point, object pose), solves a
small Jacobian IK that reaches a target position while aiming the tool +z axis
at a direction, drives the actuators, and toggles the grasp weld.

Convention: the tool approach axis is the pinch site's local +z axis. At the
home pose it points straight down (ready to grasp an object on the table).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .config_loader import Config


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class TrackController:
    def __init__(self, model: "mujoco.MjModel", cfg: Config):
        self.m = model
        self.cfg = cfg
        self.scratch = mujoco.MjData(model)  # used by IK so the main sim is untouched

        nid = lambda t, n: mujoco.mj_name2id(model, t, n)
        self.joint_names = cfg.joint_names
        self.qadr = np.array([model.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in self.joint_names])
        self.dofadr = np.array([model.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in self.joint_names])
        self.lo = cfg.arr("robot", "joint_limits_lower")
        self.hi = cfg.arr("robot", "joint_limits_upper")

        self.pinch = nid(mujoco.mjtObj.mjOBJ_SITE, "2f85_pinch")
        self.obj_qadr = model.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.obj_dofadr = model.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.obj_body = nid(mujoco.mjtObj.mjOBJ_BODY, "object")
        self.grip_body = nid(mujoco.mjtObj.mjOBJ_BODY, "2f85_base")

        self.arm_act = list(range(6))
        self.grip_act = nid(mujoco.mjtObj.mjOBJ_ACTUATOR, "2f85_fingers_actuator")
        self.weld_id = nid(mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # -- state readouts -------------------------------------------------------
    def pinch_pos(self, d):
        return d.site_xpos[self.pinch].copy()

    def arm_q(self, d):
        return d.qpos[self.qadr].copy()

    def object_pos(self, d):
        return d.qpos[self.obj_qadr:self.obj_qadr + 3].copy()

    def object_speed(self, d):
        return float(np.linalg.norm(d.qvel[self.obj_dofadr:self.obj_dofadr + 3]))

    def set_object_pose(self, d, pos, quat=None, vel=None):
        """Kinematically set the object pose (used by the auto driver / reset)."""
        a = self.obj_qadr
        d.qpos[a:a + 3] = pos
        if quat is not None:
            d.qpos[a + 3:a + 7] = quat
        v = self.obj_dofadr
        d.qvel[v:v + 6] = 0.0 if vel is None else np.concatenate([vel, np.zeros(3)])

    # -- IK: reach target_pos and aim tool +z along aim_dir -------------------
    def solve_ik(self, target_pos, aim_dir, q_seed,
                 pos_weight=1.0, ori_weight=1.0, iters=120, damping=1e-2):
        m, sd = self.m, self.scratch
        sd.qpos[:] = 0.0
        q = np.array(q_seed, dtype=float)
        aim = np.asarray(aim_dir, dtype=float)
        aim = aim / (np.linalg.norm(aim) + 1e-9)
        for _ in range(iters):
            sd.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, sd)
            mujoco.mj_comPos(m, sd)
            p = sd.site_xpos[self.pinch]
            R = sd.site_xmat[self.pinch].reshape(3, 3)
            e_pos = (np.asarray(target_pos) - p) * pos_weight
            e_ori = np.cross(R[:, 2], aim) * ori_weight   # align tool +z with aim
            if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(e_ori) < 1e-3:
                break
            mujoco.mj_jacSite(m, sd, self._jacp, self._jacr, self.pinch)
            J = np.vstack([self._jacp[:, self.dofadr] * pos_weight,
                           self._jacr[:, self.dofadr] * ori_weight])
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, np.concatenate([e_pos, e_ori]))
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    # -- grasp via weld -------------------------------------------------------
    def set_grasp(self, d, on: bool):
        if self.weld_id < 0:
            return
        if on:
            # store the current object pose in the gripper frame so the weld
            # holds it where it currently sits (no snapping)
            p1, q1 = d.xpos[self.grip_body], d.xquat[self.grip_body]
            p2, q2 = d.xpos[self.obj_body], d.xquat[self.obj_body]
            R1 = d.xmat[self.grip_body].reshape(3, 3)
            data = self.m.eq_data[self.weld_id]
            data[0:3] = 0.0
            data[3:6] = R1.T @ (p2 - p1)
            data[6:10] = _quat_mul(_quat_conj(q1), q2)
            data[10] = 1.0
            d.eq_active[self.weld_id] = 1
        else:
            d.eq_active[self.weld_id] = 0

    # -- actuator commands ----------------------------------------------------
    def command_arm(self, d, q_des):
        d.ctrl[self.arm_act] = np.clip(q_des, self.lo, self.hi)

    def command_gripper(self, d, close: bool):
        d.ctrl[self.grip_act] = (float(self.cfg.get("gripper", "close_ctrl", default=255.0))
                                 if close else
                                 float(self.cfg.get("gripper", "open_ctrl", default=0.0)))

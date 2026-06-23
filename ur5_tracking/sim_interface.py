"""SimInterface: RobotInterface backed by a live MuJoCo MjData object."""
from __future__ import annotations

import mujoco
import numpy as np

from .config_loader import Config
from .robot_interface import RobotInterface


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


class SimInterface(RobotInterface):
    """Reads state from and writes commands to a MuJoCo MjData instance."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: Config):
        self.m = model
        self.d = data

        nid = lambda t, n: mujoco.mj_name2id(model, t, n)
        joint_names = cfg.joint_names
        self.qadr = np.array([model.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, j)]
                               for j in joint_names])
        self.lo = cfg.arr("robot", "joint_limits_lower")
        self.hi = cfg.arr("robot", "joint_limits_upper")

        self.pinch      = nid(mujoco.mjtObj.mjOBJ_SITE,     "2f85_pinch")
        self.obj_qadr   = model.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.obj_dofadr = model.jnt_dofadr [nid(mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self.obj_body   = nid(mujoco.mjtObj.mjOBJ_BODY,     "object")
        self.grip_body  = nid(mujoco.mjtObj.mjOBJ_BODY,     "2f85_base")
        self.arm_act    = list(range(6))
        self.grip_act   = nid(mujoco.mjtObj.mjOBJ_ACTUATOR, "2f85_fingers_actuator")
        self.weld_id    = nid(mujoco.mjtObj.mjOBJ_EQUALITY,  "grasp_weld")

        self.close_ctrl = float(cfg.get("gripper", "close_ctrl", default=255.0))
        self.open_ctrl  = float(cfg.get("gripper", "open_ctrl",  default=0.0))

    # -- RobotInterface -------------------------------------------------------
    def get_pinch_pos(self) -> np.ndarray:
        return self.d.site_xpos[self.pinch].copy()

    def get_arm_q(self) -> np.ndarray:
        return self.d.qpos[self.qadr].copy()

    def get_object_pos(self) -> np.ndarray:
        return self.d.qpos[self.obj_qadr:self.obj_qadr + 3].copy()

    def get_object_speed(self) -> float:
        return float(np.linalg.norm(self.d.qvel[self.obj_dofadr:self.obj_dofadr + 3]))

    def command_arm(self, q_des: np.ndarray) -> None:
        self.d.ctrl[self.arm_act] = np.clip(q_des, self.lo, self.hi)

    def command_gripper(self, close: bool) -> None:
        self.d.ctrl[self.grip_act] = self.close_ctrl if close else self.open_ctrl

    def set_grasp(self, on: bool) -> None:
        if self.weld_id < 0:
            return
        if on:
            p1 = self.d.xpos [self.grip_body]
            q1 = self.d.xquat[self.grip_body]
            p2 = self.d.xpos [self.obj_body]
            q2 = self.d.xquat[self.obj_body]
            R1 = self.d.xmat [self.grip_body].reshape(3, 3)
            eq = self.m.eq_data[self.weld_id]
            eq[0:3]  = 0.0
            eq[3:6]  = R1.T @ (p2 - p1)
            eq[6:10] = _quat_mul(_quat_conj(q1), q2)
            eq[10]   = 1.0
            self.d.eq_active[self.weld_id] = 1
        else:
            self.d.eq_active[self.weld_id] = 0

    def get_time(self) -> float:
        return self.d.time

    # -- sim-only (not in RobotInterface) ------------------------------------
    def set_object_pose(self, pos, quat=None, vel=None) -> None:
        """Kinematically place the object (used by AutoObjectDriver and setup)."""
        a = self.obj_qadr
        self.d.qpos[a:a + 3] = pos
        if quat is not None:
            self.d.qpos[a + 3:a + 7] = quat
        v = self.obj_dofadr
        self.d.qvel[v:v + 6] = 0.0 if vel is None else np.concatenate([vel, np.zeros(3)])

"""Low-level control for the track / pick / return task.

TrackController solves Jacobian IK to reach a Cartesian target while aiming
the tool +z axis at a direction, then delegates all state reads and actuator
writes to a RobotInterface (sim or hardware).

Convention: the tool approach axis is the pinch site's local +z axis.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .config_loader import Config
from .robot_interface import RobotInterface


class TrackController:
    def __init__(self, model: "mujoco.MjModel", cfg: Config, iface: RobotInterface):
        self.m = model
        self.cfg = cfg
        self.iface = iface
        self.scratch = mujoco.MjData(model)  # IK only — never stepped

        nid = lambda t, n: mujoco.mj_name2id(model, t, n)
        self.joint_names = cfg.joint_names
        self.qadr = np.array([model.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, j)]
                               for j in self.joint_names])
        self.dofadr = np.array([model.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, j)]
                                 for j in self.joint_names])
        self.lo = cfg.arr("robot", "joint_limits_lower")
        self.hi = cfg.arr("robot", "joint_limits_upper")
        self.pinch = nid(mujoco.mjtObj.mjOBJ_SITE, "2f85_pinch")

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # -- delegated state readouts ---------------------------------------------
    def pinch_pos(self) -> np.ndarray:    return self.iface.get_pinch_pos()
    def arm_q(self) -> np.ndarray:        return self.iface.get_arm_q()
    def object_pos(self) -> np.ndarray:   return self.iface.get_object_pos()
    def object_speed(self) -> float:      return self.iface.get_object_speed()
    def get_time(self) -> float:          return self.iface.get_time()

    # -- delegated actuator commands ------------------------------------------
    def command_arm(self, q_des) -> None:           self.iface.command_arm(q_des)
    def command_gripper(self, close: bool) -> None: self.iface.command_gripper(close)
    def set_grasp(self, on: bool) -> None:          self.iface.set_grasp(on)

    # -- sim-only passthrough (AutoObjectDriver; never called in hardware mode)
    def set_object_pose(self, pos, quat=None, vel=None) -> None:
        self.iface.set_object_pose(pos, quat=quat, vel=vel)

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

"""HardwareInterface: Phase 2 stub.

Will wrap ur-rtde (UR5e), pyrobotiqgripper (Robotiq 2F-85 over USB-RS485),
and ArUco-based object pose estimation (opencv-contrib-python).

Install when ready:
    pip install ur-rtde pyrobotiqgripper minimalmodbus opencv-contrib-python
"""
from __future__ import annotations

from .robot_interface import RobotInterface


class HardwareInterface(RobotInterface):
    """Phase 2 placeholder — not yet implemented."""

    def __init__(self, model, cfg):
        raise NotImplementedError(
            "HardwareInterface is not yet implemented. "
            "Run without --hardware to use the simulation."
        )

    def connect(self):            raise NotImplementedError
    def disconnect(self):         raise NotImplementedError
    def get_pinch_pos(self):      raise NotImplementedError
    def get_arm_q(self):          raise NotImplementedError
    def get_object_pos(self):     raise NotImplementedError
    def get_object_speed(self):   raise NotImplementedError
    def command_arm(self, q):     raise NotImplementedError
    def command_gripper(self, c): raise NotImplementedError
    def set_grasp(self, on):      raise NotImplementedError
    def get_time(self):           raise NotImplementedError

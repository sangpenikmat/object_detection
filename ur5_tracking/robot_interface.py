"""Abstract interface between the FSM/controller and the underlying platform.

Implement SimInterface (backed by MjData) for simulation and
HardwareInterface (backed by ur-rtde + gripper + vision) for real hardware.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RobotInterface(ABC):

    @abstractmethod
    def get_pinch_pos(self) -> np.ndarray:
        """End-effector (pinch site) XYZ in robot base frame, shape (3,)."""

    @abstractmethod
    def get_arm_q(self) -> np.ndarray:
        """Current joint angles in radians, shape (6,)."""

    @abstractmethod
    def get_object_pos(self) -> np.ndarray:
        """Object XYZ in robot base frame, shape (3,)."""

    @abstractmethod
    def get_object_speed(self) -> float:
        """Scalar linear speed of the object (m/s)."""

    @abstractmethod
    def command_arm(self, q_des: np.ndarray) -> None:
        """Send desired joint angles (6,) to the arm."""

    @abstractmethod
    def command_gripper(self, close: bool) -> None:
        """Open or close the gripper."""

    @abstractmethod
    def set_grasp(self, on: bool) -> None:
        """Enable/disable grasp coupling. No-op on hardware."""

    @abstractmethod
    def get_time(self) -> float:
        """Elapsed time in seconds (sim time or wall clock)."""

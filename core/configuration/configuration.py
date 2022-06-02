from __future__ import annotations
from typing import Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class Joint_Controller_PID_Values:
    kp: float
    ki: float
    kd: float

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> Joint_Controller_PID_Values:
        return cls(**dictionary)


@dataclass
class Cartesian_Controller_PID_Values:
    kp: float
    ki: float
    kd: float

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> Cartesian_Controller_PID_Values:
        return cls(**dictionary)


@dataclass
class RobotConfiguration:
    parking_position: np.ndarray

    controller_frequency: Tuple[float, float, float]
    joint_limits: Tuple[float, float, float]
    joint_controller_pid: Joint_Controller_PID_Values
    cartesian_controller_pid: Cartesian_Controller_PID_Values

    enable_teleoperation: bool
    teleoperation_mode: str

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> RobotConfiguration:
        return cls(**dictionary)

from dataclasses import dataclass
import os
from omegaconf import OmegaConf

import numpy as np
from time import sleep

import pinocchio as pin

from core.configuration import RobotConfiguration
from core.motion import MotionPlanner
from core.robot_simulator import NYUFingerSimulator
from core.teleoperator import NumStick
from core.controller import Controller
from core.kinematics import ForwardKinematics


@dataclass
class RobotMemory:
    current_pose: np.ndarray
    latest_stable_joint_states: np.ndarray


class Robot(RobotMemory):
    def __init__(self, yaml_config_path: str) -> None:

        # Init. configuration
        config_path = os.path.abspath(yaml_config_path)

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Could not find the specified config. file at: {config_path}")

        config_dictionary = OmegaConf.load(config_path)
        configuration = RobotConfiguration.from_dictionary(config_dictionary)

        # Load the .urdf files
        self._package_dir = os.path.abspath(os.getcwd())
        self._urdf_path = self._package_dir + '/model/fingeredu.urdf'

        # Create Models from the .urdf files
        self._model, self._collision_model, self._visual_model = pin.buildModelsFromUrdf(self._urdf_path,
                                                                                         self._package_dir)
        self._model_data = self._model.createData()
        self._collision_model.addAllCollisionPairs()
        self._collision_data = pin.GeometryData(self._collision_model)

        # Display the models in viewer
        self._simulator = NYUFingerSimulator(self._package_dir, self._urdf_path)
        self._eoat_id: int = self._model.getFrameId('finger_tip_link')

        # Read Initial Joint Values
        _init_q, _init_v = self._simulator.get_state()
        self._latest_stable_joint_states = _init_q
        self._current_pose = ForwardKinematics(_init_q, _init_v)[0]

        # Assigning robot features
        self.joint_limits = configuration.joint_limits
        self.parking_poistion = configuration.parking_position

        # Set Controller
        self.controller = Controller(configuration,
                                     self._simulator,
                                     self._model,
                                     self._model_data,
                                     configuration.joint_controller_pid,
                                     configuration.cartesian_controller_pid)

        # Init. Motion Planner
        self.Motion = MotionPlanner(self._model,
                                    self._model_data,
                                    self.controller,
                                    self._simulator,
                                    self.joint_limits)

        # Assign teleoperator
        if configuration.enable_teleoperation:
            numpad_as_teleoperator = NumStick()
            numpad_as_teleoperator.run()

    def Wait(self, seconds: float) -> None:
        """
        Description:
            1. Makes the robot wait for few seconds

        Args:
            seconds -> Float: Seconds
        """
        sleep(seconds)

    def Move_P2P(self, goal_pos: np.ndarray, goal_vel: np.ndarray = None, mode='linear') -> None:

        if mode == 'linear':
            target_vel = 0.5 if goal_vel is None else goal_vel
            self.Motion.LinearPlanner(goal_pos, target_vel)

        elif mode == 'joint':
            target_vel = np.deg2rad(90) if goal_vel is None else goal_vel
            self.Motion.JointPlanner(goal_pos, target_vel)

        elif mode == 'circular':
            target_vel = 0.5 if goal_vel is None else goal_vel
            self.Motion.CircularPlanner(goal_pos, target_vel)

    def Exit(self) -> None:
        """
        Description:
            1. Exit the Robot simulator.
        """
        print('Exiting ...')
        self.controller.engage = False
        self.controller.stop()

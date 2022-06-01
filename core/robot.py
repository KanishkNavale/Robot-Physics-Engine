import os
from omegaconf import OmegaConf

import numpy as np
from time import sleep

import pinocchio as pin

from core.configuration import RobotConfiguration
from core.motion import MotionPlanner
from core.robot_simulator import NYUFingerSimulator
from core.teleoperator import NumStick


class Robot:
    def __init__(self, yaml_config_path: str) -> None:

        # Init. configuration
        config_path = os.path.abspath(yaml_config_path)

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Could not find the specified config. file at: {config_path}")

        config_dictionary = OmegaConf.load(config_path)
        self.config = RobotConfiguration.from_dictionary(config_dictionary)

        # Load the .urdf files
        self.package_dir = os.path.abspath(os.getcwd())
        self.urdf_path = self.package_dir + '/model/fingeredu.urdf'

        # Create Models from the .urdf files
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(self.urdf_path, self.package_dir)
        self.model_data = self.model.createData()
        self.collision_model.addAllCollisionPairs()
        self.collision_data = pin.GeometryData(self.collision_model)

        # Display the models in viewer
        self.simulator = NYUFingerSimulator(self.package_dir, self.urdf_path)
        self.eoat_id: int = self.model.getFrameId('finger_tip_link')

        # Read Initial Joint Values
        init_q, init_v = self.simulator.get_state()

        # Initial Position
        configuration.pose = self.ForwardKinematics(init_q, init_v)[0]

        # PD Controller
        self.control_freq = 0.001
        self.PD = SimPD(self.control_freq,
                        self.simulator,
                        self.model,
                        self.model_data,
                        self.ForwardKinematics,
                        self.Jacobian,
                        self.TimeDerivativeJacobian,
                        9500, 35, 1.5,
                        18000, 100, 50)

        # Direct Joint Limits
        J1_limits = np.pi / 2
        J2_limits = np.pi
        J3_limits = np.pi
        self.JointLimits = np.array([J1_limits, J2_limits, J3_limits])

        # Init. Motion Planner
        self.Motion = MotionPlanner(self.model,
                                    self.model_data,
                                    self.PD,
                                    self.simulator,
                                    self.ForwardKinematics,
                                    self.Jacobian,
                                    self.JointLimits)

        # Feature Configuration Spaces
        self.StablePose = init_q
        self.parking_pose = np.array([0, -np.pi / 4, np.pi / 4])

        TeleKeyboard = NumStick(self.Motion.LinearPlanner)
        TeleKeyboard.run()

    def Wait(self, seconds: float) -> None:
        """
        Description:
            1. Makes the robot wait for few seconds

        Args:
            seconds -> Float: Seconds
        """
        sleep(seconds)

    def Move_P2P(self, goal_pos: np.ndarray, goal_vel: np.ndarray = None, mode='linear') -> None:
        """
        Description:
            1. Moves the robot towards the target cartesian point.
            2. The code is based of optimization.
            source: "notes/03-kinematics-part2.pdf"

        Args:
            goal_pos -> np.array (3,) : Target Position in Cartesian sSpace.

        Keyword Args:
            1. mode -> string: Interpolation mode contains 'linear',
                               'joint' & 'circular'. Defaults to 'linear'.
            2. C -> list(2, (3,)): To be used when 'circular' mode is used.
                                   Estimates a 3-Point Circular Trajectory.
                                   Defaults to None.

        Returns:
            list(n, (3,)): List of the EOAT(TCP) logged positions.
        """

        if mode == 'linear':
            if goal_vel is None:
                target_vel = 0.5
            else:
                target_vel = goal_vel

            # Invoke Linear Planner
            self.StablePose = self.Motion.LinearPlanner(goal_pos, target_vel)

        elif mode == 'joint':
            if goal_vel is None:
                target_vel = np.deg2rad(90)
            else:
                target_vel = goal_vel

            # Invoke Joint Planner
            self.StablePose = self.Motion.JointPlanner(goal_pos, target_vel)

        elif mode == 'circular':
            if goal_vel is None:
                target_vel = 0.5
            else:
                target_vel = goal_vel

            # Invoke Joint Planner
            self.StablePose = self.Motion.CircularPlanner(goal_pos, target_vel)

    def Exit(self) -> None:
        """
        Description:
            1. Exit the Robot simulator.
        """
        print('Exiting ...')
        self.PD.engage = False
        self.PD.controller.join()

from typing import Any, List

import numpy as np
from scipy.linalg import pinv
import pinocchio as pin

from threading import Thread
from time import sleep

from core.robot_simulator import NYUFingerSimulator
from core.kinematics import ForwardKinematics, FrameJacobian
from core.configuration import RobotConfiguration


class Controller:
    def __init__(self,
                 robot_configuration: RobotConfiguration,
                 simulator: NYUFingerSimulator,
                 model: Any,
                 model_data: Any):

        # Initialize Control based Terms
        self._frequency = robot_configuration.controller_frequency

        self._ki = robot_configuration.joint_controller_pid.ki
        self._ki_copy = robot_configuration.joint_controller_pid.ki

        self._li = robot_configuration.cartesian_controller_pid.ki
        self._li_copy = robot_configuration.cartesian_controller_pid.ki
        self._integral_error_history: List[float] = []

        self._joint_controller_pid = robot_configuration.joint_controller_pid
        self._cartesian_controller_pid = robot_configuration.cartesian_controller_pid

        self._current_space_routine = 'joint_space'

        # Initialize the State Readers
        self._simulator = simulator

        self._joint_position = self._simulator.get_state()[0]
        self._joint_velocity = self._simulator.get_state()[1]

        self.target_position = self._joint_position
        self.target_velocity = self._joint_velocity

        self._previous_target_position = np.ones(3)

        # Initialize the Model based Compute Terms
        self._model = model
        self._data = model_data

        # Thread Management
        self._engage = True
        self._controller = Thread(target=self._joint_control)
        self._controller.start()

    def stop(self) -> None:
        self._controller.join()

    def _joint_control(self):
        while self._engage:
            # Reset for the Integral History
            if not np.allclose(self.target_position, self._previous_target_position):
                self._integral_error_history = []

            # Read the states
            self._joint_position = self._simulator.get_state()[0]
            self._joint_velocity = self._simulator.get_state()[1]

            # Control's Offline Term Estimates
            M = pin.crba(self._model, self._data, self._joint_position)
            N = pin.nonLinearEffects(self._model,
                                     self._data,
                                     self._joint_position,
                                     self._joint_velocity)

            tau = (self._joint_controller_pid.kp * (self.target_position - self._joint_position) +
                   self._joint_controller_pid.kd * (self.target_velocity - self._joint_velocity) +
                   self._joint_controller_pid.ki * np.sum(self._integral_error_history))

            tau: np.ndarray = M @ tau + N
            clipped_TAU = np.clip(tau, -10, 10)

            self._simulator.send_joint_torque(clipped_TAU)
            self._simulator.step()

            # PID: Integral Anti-Windup
            error = self.target_position - self._joint_position
            if not np.allclose(tau, clipped_TAU):
                sign_tau = np.sign(tau)
                sign_error = np.sign(error)
                for i in range(sign_tau.shape[0]):
                    if sign_tau[i] == sign_error[i]:
                        self._ki = 0
                        break
            else:
                self._ki = self._ki_copy
                self._integral_error_history.append(error)

            # Store the Desired Position
            self._previous_target_position = self.target_position

            sleep(self._frequency)

    def _taskspace_control(self):
        while self.engage:
            # Reset for the Integral History
            if not np.allclose(self.target_position, self._previous_target_position):
                self._integral_error_history = []

            # Read the states
            self._joint_position = self._simulator.get_state()[0]
            self._joint_velocity = self._simulator.get_state()[1]
            y, v = ForwardKinematics(self._joint_position, self._joint_velocity)

            # Control's Offline Term Estimates
            M = pin.crba(self._model, self._data, self._joint_position)
            N = pin.nonLinearEffects(self._model,
                                     self._data,
                                     self._joint_position,
                                     self._joint_velocity)

            tau = (self._joint_controller_pid.kp * (self.target_position - y) +
                   self._joint_controller_pid.kp * (self.target_velocity - v) +
                   self._li * np.sum(self._integral_error_history))

            # Build Online Control Terms
            J = FrameJacobian(self._joint_position)
            DJ = J  # Naive Method as Pinocchio for DJ is not stable.
            JM = M @ pinv(J)

            # Compute & Send Torques
            tau: np.ndarray = (JM @ tau) + N - 2 * (JM @ DJ @ self._joint_velocity)
            clipped_tau = np.clip(tau, -10, 10)

            self._simulator.send_joint_torque(clipped_tau)
            self._simulator.step()

            # PID: Integral Anti-Windup
            error = self.target_position - y
            if not np.allclose(tau, clipped_tau):
                sign_tau = np.sign(tau)
                sign_error = np.sign(error)
                for i in range(sign_tau.shape[0]):
                    if sign_tau[i] == sign_error[i]:
                        self._li = 0
                        break
            else:
                self._li = self._li_copy
                self._integral_error_history.append(error)

            # Store the Desired Position
            self._previous_target_position = self.target_position

            sleep(self._frequency)

    def control_loop(self, routine):
        self._current_space_routine = routine
        if self._current_space_routine == 'joint_space':

            if self.prev_routine != self._current_space_routine:
                self._engage = False
                self._controller.join()

                self._engage = True
                self._controller = Thread(target=self._joint_control)
                sleep(0.2)

                self._controller.start()
                self.prev_routine = 'joint_space'

        if self._current_space_routine == 'cartesian_space':
            if self.prev_routine != self._current_space_routine:
                self.engage = False
                self.controller.join()

                self.engage = True
                self.controller = Thread(target=self._taskspace_control)
                sleep(0.2)

                self.controller.start()
                self.prev_routine = 'cartesian_space'

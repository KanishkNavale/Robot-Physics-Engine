from typing import Any
from copy import deepcopy

import numpy as np
import pinocchio as pin
from threading import Thread

from core.robot_simulator import NYUFingerSimulator
from core.robot import Robot


class Controller:
    def __init__(self,
                 frequency: float,
                 simulator: NYUFingerSimulator,
                 model: Any,
                 model_data: Any,
                 kinematics_solver: Robot.ForwardKinematics,
                 jacobian_solver: Robot.Jacobian,
                 hessian_solver: Robot.TimeDerivativeJacobian):

        # Initialize Control based Terms
        self.freq = frequency
        self.Ki_copy = Ki
        self.Li_copy = Li
        self.log = []
        self.routine = 1
        self.prev_routine = 0

        # Initialize the State Readers
        self.simulator = simulator
        self.joint_positions = self.simulator.get_state()[0]
        self.joint_velocities = self.simulator.get_state()[1]
        self.desired_position = deepcopy(self.joint_positions)
        self.desired_velocity = deepcopy(self.joint_velocities)
        self.prev_DesiredPosition = np.ones(3)

        # Initialize the Model based Compute Terms
        self.model = model
        self.data = model_data
        self.kinematics_solver = kinematics_solver
        self.jacobian_solver = jacobian_solver
        self.hessian_solver = hessian_solver

        # Thread Management
        self.engage = True
        self.controller = Thread(target=self.joint_control)
        self.controller.start()

    def joint_control(self):
        while self.engage:
            # Reset for the Integral History
            if not np.array_equal(self.desired_position,
                                  self.prev_DesiredPosition):
                self.log = []

            # Read the states
            self.joint_positions = self.simulator.get_state()[0]
            self.joint_velocities = self.simulator.get_state()[1]

            # Control's Offline Term Estimates
            M = pin.crba(self.model, self.data, self.joint_positions)
            N = pin.nonLinearEffects(self.model, self.data,
                                     self.joint_positions,
                                     self.joint_velocities)

            tau = (self.Kp * (self.desired_position - self.joint_positions) +
                   self.Kd * (self.desired_velocity - self.joint_velocities) +
                   self.Ki * np.sum(self.log))

            self.TAU = M @ tau + N
            self.clipped_TAU = np.clip(self.TAU, -10, 10)

            self.simulator.send_joint_torque(self.clipped_TAU)
            self.simulator.step()

            # PID: Integral Anti-Windup
            error = self.desired_position - self.joint_positions
            if not np.array_equal(self.TAU, self.clipped_TAU):
                sign_tau = np.sign(self.TAU)
                sign_error = np.sign(error)
                for i in range(sign_tau.shape[0]):
                    if sign_tau[i] == sign_error[i]:
                        self.Ki = 0
                        break
            else:
                self.Ki = self.Ki_copy
                self.log.append(error)

            # Store the Desired Position
            self.prev_DesiredPosition = self.desired_position

            sleep(self.freq)

    def taskspace_control(self):
        while self.engage:
            # Reset for the Integral History
            if not np.array_equal(self.desired_position,
                                  self.prev_DesiredPosition):
                self.log = []

            # Read the states
            self.joint_positions = self.simulator.get_state()[0]
            self.joint_velocities = self.simulator.get_state()[1]
            y, v = self.FK_Solver(self.joint_positions,
                                  self.joint_velocities)

            # Control's Offline Term Estimates
            M = pin.crba(self.model, self.data, self.joint_positions)
            N = pin.nonLinearEffects(self.model, self.data,
                                     self.joint_positions,
                                     self.joint_velocities)

            tau = (self.Lp * (self.desired_position - y) +
                   self.Ld * (self.desired_velocity - v) +
                   self.Li * np.sum(self.log))

            # Build Online Control Terms
            J = self.Jacobian_Solver(self.joint_positions)
            DJ = J  # Naive Method as Pinocchio for DJ is not stable.
            JM = M @ linalg.pinv(J)

            # Compute & Send Torques
            self.TAU = (JM @ tau) + N - 2 * (JM @ DJ @ self.joint_velocities)
            self.clipped_TAU = np.clip(self.TAU, -10, 10)

            self.simulator.send_joint_torque(self.clipped_TAU)
            self.simulator.step()

            # PID: Integral Anti-Windup
            error = self.desired_position - y
            if not np.array_equal(self.TAU, self.clipped_TAU):
                sign_tau = np.sign(self.TAU)
                sign_error = np.sign(error)
                for i in range(sign_tau.shape[0]):
                    if sign_tau[i] == sign_error[i]:
                        self.Li = 0
                        break
            else:
                self.Li = self.Li_copy
                self.log.append(error)

            # Store the Desired Position
            self.prev_DesiredPosition = self.desired_position

            sleep(self.freq)

    def control_loop(self, routine):
        self.routine = routine
        if self.routine == 1:
            # Check for Change in routine
            if self.prev_routine != self.routine:
                # Stop the current controller
                self.engage = False
                self.controller.join()
                self.engage = True
                self.controller = Thread(target=self.joint_control)
                sleep(0.2)
                self.controller.start()
                self.prev_routine = self.routine

        if self.routine == 2:
            # Check for Change in routine
            if self.prev_routine != self.routine:
                # Stop the current controller
                self.engage = False
                self.controller.join()
                # Start a new controller
                self.engage = True
                self.controller = Thread(target=self.taskspace_control)
                sleep(0.2)
                self.controller.start()
                self.prev_routine = self.routine

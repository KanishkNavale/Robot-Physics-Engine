#############################################################################
# DEVELOPED BY KANISHK                                          #############
# THIS SCRIPT CONSTRUCTS A MOTION PLANNER FOR ROBOT.            #############
#############################################################################

# Library Imports
import numpy as np
from scipy import linalg
from copy import deepcopy
from time import sleep
import pinocchio as pin
from collections import deque
from threading import Thread
import memory


#############################################################################
# PID CONTROLLER CLASS                                          #############
#############################################################################
class SimPD:
    def __init__(self, freq, simulator, model, data,
                 FK_Solver, Jacobian_Solver, DJacobian_Solver,
                 Kp, Kd, Ki, Lp, Ld, Li):
        # Initialize Control based Terms
        self.freq = freq
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.Ki_copy = Ki
        self.Lp = Lp
        self.Ld = Ld
        self.Li = Li
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
        self.data = data
        self.FK_Solver = FK_Solver
        self.Jacobian_Solver = Jacobian_Solver
        self.DJacobian_Solver = DJacobian_Solver

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
            self.TAU = (JM @ tau) + N - (JM @ DJ @ self.joint_velocities)
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


#############################################################################
# Motion PLANNER                                                #############
#############################################################################
class MotionPlanner:
    def __init__(self, model, model_data,
                 controller, simulator,
                 FK_solver, Jacobian_solver, JointLimits):

        self.model = model
        self.model_data = model_data
        self.controller = controller
        self.simulator = simulator
        self.FK_solver = FK_solver
        self.JointLimits = JointLimits
        self.Jacobian_solver = Jacobian_solver
        self.StablePos_log = deque(maxlen=5)

        # Fill the buffer
        for i in range(self.StablePos_log.maxlen):
            q = self.GetStates()[0]
            self.StablePos_log.append(q)

    def GetStates(self):
        """
        Description,
        1. Reads the state of robot.
        2. Updates the pose in the global memory.

        Returns:
            type=np.float32, shape=(3,): joint_positions.
            type=np.flaot32, shape=(3,): joint_velocities.
        """
        q, qdot = np.float32(self.simulator.get_state())

        memory.pose = deepcopy(self.FK_solver(q, qdot))[0]
        return q, qdot

    def CheckJoints(self, q):
        """
        Description,
        1. Check joints for joint limits.

        Args:
            q (type=np.float32, shape=(3,)): Joint Configuration.

        Returns:
            exception: Joint Limit Exception.
        """
        if q[0] < -self.JointLimits[0] or \
                q[0] > self.JointLimits[0]:
            raise Exception("J1 Joint Limit Reached")

        elif q[1] < -self.JointLimits[1] or \
                q[1] > self.JointLimits[1]:
            raise Exception("J2 Joint Limit Reached")

        elif q[2] < -self.JointLimits[2] or \
                q[2] > self.JointLimits[2]:
            raise Exception("J3 Joint Limit Reached")

    def SetStates(self, target_pos, target_vel, method):
        """
        Description,
        1. Sends the target state to PID Planner.

        Args:
            target_q (type=np.flaot32, shape=(3,)): Target Joint Position.
            target_qdot (type=np.flaot32, shape=(3,)): Target Joint Velocity.
            routine(int): Switches in between Cartesian & Joint Control.
        """
        self.controller.control_loop(method)
        self.controller.desired_position = target_pos
        self.controller.desired_velocity = target_vel

    def ComputeTime(self, init_pos, final_pos, init_vel, final_vel):
        """
        Description,
        1. Computes the physical time between two positions based on
           initial & final velocities.

        Args:
            init_pos (type=np.float32, shape=(3,)): Initial Position.
            final_pos (type=np.float32, shape=(3,)): Final Position.
            init_vel (type=np.float32, shape=(3,)): Initial Velociry.
            final_vel (type=np.float32, shape=(3,)): Final Velocity.

        Returns:
            float: time.
        """
        time = np.divide(1 * (final_pos - init_pos),
                         (final_vel - init_vel))
        time = np.max(np.abs(time))
        return time

    def JointPlanner(self, target_q, target_v):
        """
        Description,
        1. Motion Planner for the joint based motion.

        Args:
            target_q (type=np.float32, shape=(3,)): Target Joint Position.
            target_v (type=np.float32, shape=(3,)): Target Joint Velocity.
        """
        # Initialize Target Velocity,
        max_speed = np.deg2rad(360)
        min_speed = np.deg2rad(25)
        if np.isscalar(target_v):
            if np.abs(target_v) > max_speed:
                target_v = np.clip(target_v, -max_speed, max_speed)
            if np.abs(target_v) < min_speed:
                target_v = np.clip(target_v, -min_speed, min_speed)
            if np.abs(target_v) == 0.0:
                target_v = min_speed
            target_v = target_v * np.ones(3)
        else:
            if np.any(np.abs(target_v) > max_speed):
                target_v = np.clip(target_v, -max_speed, max_speed)
            if np.any(np.abs(target_v) < min_speed):
                target_v = np.clip(target_v, -min_speed, min_speed)

        # Compute the max. distance & time
        q, qdot = self.GetStates()
        init_q = deepcopy(q)
        time = self.ComputeTime(q, target_q, qdot, target_v)

        # Initialize Time Factors
        tau = self.controller.freq
        N = int(time / tau)
        T = N * tau

        try:
            for i in range(1, N):

                t = i * tau

                # Augment targets
                augmented_q = init_q + (t / T) * (target_q - init_q)

                # Check Joint Limits
                self.CheckJoints(augmented_q)

                # Set the States
                self.SetStates(augmented_q, target_v, method=1)

                # Refresh the States
                self.GetStates()

                sleep(tau)

        except Exception as e:
            print(e)

    def LinearPlanner(self, goal_pos, goal_vel=0.5):
        """
        Description,
        1. Motion Planner for Cartesian Space.

        Args:
            goal_pos (type=np.float32, shape=(3,)): Cartesian Target Position.
            goal_vel (float, optional): Cartesian Target Velocity. Default=0.5.
        """
        # Clip the velocity
        if goal_vel > 1 or goal_vel < 0.1:
            goal_vel = np.clip(goal_vel, 0.1, 1)
        goal_vel = goal_vel * np.ones(3)

        # Calculate time for transition
        q, qdot = self.GetStates()
        init_pos, init_vel = self.FK_solver(q, qdot)
        time = self.ComputeTime(init_pos, goal_pos, init_vel, goal_vel)

        # Initialize Time Factors
        tau = self.controller.freq
        N = int(time / tau)
        T = N * tau

        try:
            for i in range(1, N):

                t = i * tau

                # Augment targets
                augmented_y = init_pos + (t / T) * (goal_pos - init_pos)

                # Check Joint Limits
                self.CheckJoints(augmented_y)

                # Set the States
                self.SetStates(augmented_y, goal_vel, method=2)

                # Refresh the States
                self.GetStates()

                sleep(tau)

        except Exception as e:
            print(e)

    def CircularPlanner(self, goal_pos, goal_vel):
        """
        Description,
        1. Motion Planner for Cartesian Circular Path.

        Args:
            goal_pos (type=np.float32, shape=(1,3)): 3-Points in
                                                     Cartesian Space to fit &
                                                     generate Circular Path.
            goal_vel (np.float32): Cartesian Target Velocity.
        """
        # Calculate Parameters of the Circle
        P1 = goal_pos[0]
        P2 = goal_pos[1]
        P3 = goal_pos[2]
        C = (P1 + P2 + P3) / 3

        # Compute Normal Vector to Plane
        n = np.cross((P2 - P1), (P3 - P1))
        norm_n = np.linalg.norm(n)
        if norm_n == 0:
            n = 0
        else:
            n = n / np.linalg.norm(n)

        # Pick a Point on a Plane
        Q = C + np.cross(n, (P1 - C))

        theta = np.arange(0, 2 * np.pi, 1 / 360)
        x = C[0] + np.cos(theta) * (P1[0] - C[0]) + \
            np.sin(theta) * (Q[0] - C[0])
        y = C[1] + np.cos(theta) * (P1[1] - C[1]) + \
            np.sin(theta) * (Q[1] - C[1])
        z = C[2] + np.cos(theta) * (P1[2] - C[2]) + \
            np.sin(theta) * (Q[2] - C[2])

        # Traverse the Points
        for j in range(len(x)):
            y_target = np.array([x[j], y[j], z[j]])
            self.LinearPlanner(y_target, goal_vel)

#############################################################################
# DEVELOPED BY KANISHK                                          #############
# THIS SCRIPT CONSTRUCTS A ROBOT CLASS.                         #############
#############################################################################

# Imports for System Handlers
import os

# Using OpenRobots Scripts
import pinocchio as pin

# For Manipulating Tensors
import numpy as np
from time import sleep

# 'Core' & 'Forge Imports'
from motion import MotionPlanner, SimPD
from robot_simulator import NYUFingerSimulator
from teleoperator import NumStick
import memory


##############################################################################
# ROBOT CLASS                                                    #############
##############################################################################
class Robot:
    """ Contructs a class for the 'nyu_finger' robot """

    def __init__(self, controller=None):
        """
        Description:
            1. Initiates, Builds & Starts Robot Model components.
            2. Initializes 'NYUFingerSimulator()' as Visualizer.

        Args:
            1. connect_hardware = True (default).
                                  Connects to the robot hardware if enabled
                                  else, runs as simulation.

        Keyword Args:
            1. controller = 'None': Does not initialize a joystick.
                            'keyboard': Enables joystick mode on the keyboard.
        """

        # Load the .urdf files
        self.package_dir = os.path.abspath(os.getcwd())
        self.urdf_path = self.package_dir + '/model/fingeredu.urdf'

        # Create Models from the .urdf files
        self.model, self.collision_model, self.visual_model = \
            pin.buildModelsFromUrdf(self.urdf_path, self.package_dir)

        self.collision_model.addAllCollisionPairs()

        # Create datas from models
        self.model_data = self.model.createData()
        self.collision_model.addAllCollisionPairs()
        self.collision_data = pin.GeometryData(self.collision_model)

        # Display the models in viewer
        self.simulator = NYUFingerSimulator(self.package_dir, self.urdf_path)

        # Get the FrameID of the EOAT
        self.EOAT_ID = self.model.getFrameId('finger_tip_link')

        # Get the FrameID of the Base
        self.Base_ID = self.model.getFrameId('base_link')

        # Read Initial Joint Values
        init_q, init_v = self.simulator.get_state()

        # Initial Position
        memory.pose = self.ForwardKinematics(init_q, init_v)[0]

        # PD Controller
        self.control_freq = 0.001
        self.PD = SimPD(self.control_freq, self.simulator,
                        self.model, self.model_data,
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
        self.Motion = MotionPlanner(self.model, self.model_data,
                                    self.PD, self.simulator,
                                    self.ForwardKinematics,
                                    self.Jacobian, self.JointLimits)

        # Feature Configuration Spaces
        self.StablePose = init_q
        self.parking_pose = np.array([0, -np.pi / 4, np.pi / 4])

        # Use Keyboard as a controller
        if controller == 'keyboard':
            TeleKeyboard = NumStick(self.Motion.LinearPlanner)
            TeleKeyboard.run()

    ###########################################################################
    # KINEMATICS                                                 ##############
    ###########################################################################
    def ForwardKinematics(self, q, v=np.zeros(3)):
        """
        Description:
            1. Computes the Cartesian Position & Velocity of the robot
            2. The TCP position is w.r.t. the 'tip_link' of the robot.

        Args:
            q -> np.array (3,) : Joint Positions of the robot.
            v -> np.array (3,) : Joint Velocities of the robot.

        Returns:
            np.array (3,) : Cartesian Position of the robot.
            np.array (3,) : Cartesian Velocity of the robot.
        """
        index = self.EOAT_ID
        frame = pin.ReferenceFrame.LOCAL
        pin.forwardKinematics(self.model, self.model_data, q, v)
        pin.updateFramePlacements(self.model, self.model_data)
        pos = pin.updateFramePlacement(self.model, self.model_data, index)
        vel = pin.getFrameVelocity(self.model, self.model_data, index, frame)
        return np.array(pos)[:3, -1], np.array(vel)[:3]

    def Jacobian(self, q):
        """
        Description:
            1. Computes the Jacobian Tensor of the robot.

        Args:
            q -> np.array (3,) : Joint Positions of the robot.

        Returns:
            np.array (3,3): Jacobian Tensor of the robot.
        """
        frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        pin.computeJointJacobians(self.model, self.model_data, q)
        return pin.getFrameJacobian(self.model, self.model_data,
                                    self.EOAT_ID, frame)[:3]

    def TimeDerivativeJacobian(self, q, v):
        """
        Description:
            1. Computes the Time Derivative Jacobian Tensor of the robot.

        Args:
            q -> np.array (3,) : Joint Positions of the robot.
            v -> np.array (3,) : Joint Velocities of the robot.

        Returns:
            np.array (3,3): Time Derivative Jacobian Tensor of the robot.
        """
        frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        pin.computeJointJacobians(self.model, self.model_data, q)
        pin.framesForwardKinematics(self.model, self.model_data, q)
        pin.computeJointJacobiansTimeVariation(self.model,
                                               self.model_data, q, v)
        return pin.getJointJacobianTimeVariation(self.model, self.model_data,
                                                 self.EOAT_ID, frame)[:3]

    def wait(self, seconds):
        """
        Description:
            1. Makes the robot wait for few seconds

        Args:
            seconds -> Float: Seconds
        """
        sleep(seconds)

    def Move_P2P(self, goal_pos, goal_vel=None, mode='linear'):
        """
        Description:
            1. Moves the robot towards the target cartesian point.
            2. The code is based of optimization.
            source: "notes/03-kinematics-part2.pdf"

        Args:
            goal_pos -> np.array (3,) : Target Position in Cartesian Space.

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

    def exit(self):
        """
        Description:
            1. Exit the Robot simulator.
        """
        print('Exiting ...')
        self.PD.engage = False
        self.PD.controller.join()

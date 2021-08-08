######################################################################################################################
# DEVELOPED BY KANISHK                                                                                   #############
                                                                                                         #############
# THIS SCRIPT CONSTRUCTS A ROBOT CLASS BASED ON FUNCTIONALITIES FROM THE 'FORGE'                         #############
######################################################################################################################

# Library Imports

# Imports for System Handlers
import sys
import os
from threading import Thread, Event
from multiprocessing import Process
from time import sleep, clock

# For using 'Forge' Functionalities
sys.path.append('forge')
from robot_simulator import NYUFingerSimulator

# Using OpenRobots Scripts
from nyu_finger import NYUFingerReal
import pinocchio as pin

# For Manipulating Tensors
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# For Plotting Purposes
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# For Controllers
import pygame
 

#######################################################################################################################
# Class Construct of the robot                                                                            #############
#######################################################################################################################
class Robot:
    """ Contructs a class for the 'nyu_finger' robot """

    def __init__(self, controller=None):
        """
        Description:
            1. Initiates, Builds & Starts Robot Model components and hardware.
            2. Initializes 'NYUFingerSimulator()' as Visualizer to run at the rate of 60Hz. 
            3. Runs Inverse Dynamic Compensation as parallel process (thread).
            4. Invokes 'self.offset_states()' which reads the first starting position of robot and makes it Zero.

        Keyword Args:
            1. controller = 'None': Does not initialize any joystick mode for the robot.
                            'keyboard': Enables joystick mode on the keyboard. 
        """

        # Load the .urdf files
        self.package_dir = os.path.abspath(os.getcwd())
        self.urdf_path = self.package_dir + '/model/fingeredu.urdf'

        # Create Models from the .urdf files
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(self.urdf_path, self.package_dir)
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

        # Offset Joint Values
        q, v = self.simulator.get_state()

        self.joint_offsets = q
        self.vel_offsets = v

        # Initial Position
        self.pose = self.compute_ForwardKinematics(self.joint_offsets)

        # Variables for Dynamic Compensation
        q, v = self.offset_states()
        self.set_pos = q
        self.set_vel = v

        # Refresh Display
        self.update_simulator = Thread(target=self.refresh_viz)
        self.update_simulator.start()

        # Use Keyboard as a controller
        if controller =='keyboard':
            pygame.init()
            pygame.display.set_mode((300, 300))
            self.controller = Thread(target=self.keyboard_listener)
            self.controller.start()


    ####################################################################################################################
    # Refresh Display                                                                                     ##############
    ####################################################################################################################
    def refresh_viz(self):
        """ Refreshes the display at a cycle of 60Hz """

        refresh_rate = 60 #(60FPS)
        while True:
            q, v = self.offset_states()
            self.simulator.reset_state(q, v)
            sleep(1/refresh_rate)


    ####################################################################################################################
    # Keyboard Listener                                                                                   ##############
    ####################################################################################################################
    def keyboard_listener(self):
        """
        Description:
            1. Uses the keyboard for teleoperating the robot in the cartesian space.
            2. Moves the robot with the fixed step size of 0.005m.
            3. KeyBindings,
                a. NumPad '6': +Y
                b. NumPad '4': -Y
                c. NumPad '8': -X
                d. NumPad '2': +X
                e. ArrowKey UP: +Z
                f. ArrowKey Down : -Z
        """

        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_KP6:
                        print (f'Moving Robot: Direction +Y')
                        self.pose = self.pose + np.array([0, 0.005, 0])
                        self.Move_P2P(self.pose)
                    if event.key == pygame.K_KP4:
                        print (f'Moving Robot: Direction -Y')
                        self.pose = self.pose + np.array([0, -0.005, 0])
                        self.Move_P2P(self.pose)
                    if event.key == pygame.K_KP8:
                        print (f'Moving Robot: Direction -X')
                        self.pose = self.pose + np.array([-0.005, 0.0, 0])
                        self.Move_P2P(self.pose)
                    if event.key == pygame.K_KP2:
                        print (f'Moving Robot: Direction +X')
                        self.pose = self.pose + np.array([0.005, 0.0, 0])
                        self.Move_P2P(self.pose)
                    if event.key == pygame.K_UP:
                        print (f'Moving Robot: Direction +Z')
                        self.pose = self.pose + np.array([0.0, 0.0, 0.005])
                        self.Move_P2P(self.pose)
                    if event.key == pygame.K_DOWN:
                        print (f'Moving Robot: Direction -Z')
                        self.pose = self.pose + np.array([0.0, 0.0, -0.005])
                        self.Move_P2P(self.pose)


    ####################################################################################################################
    # KINEMATICS                                                                                          ##############
    ####################################################################################################################

    def offset_states(self):
        """
        Description:
            1. Reads the initial position & velocity during the Class Initialization. 
            2. Uses the package function 'device.get_state()' to read real time position and velocity.
            3. Computes & Returns the offset position from initial reading by differencing.

        Returns:
            1. np.array (3,): Realtime Offset Position 
            2. np.array (3,): Realtime Offset Velocity
        """

        if self.hardware:
            q, v = self.device.get_state()
        else:
            q, v = self.simulator.get_state()

        return q - self.joint_offsets, v - self.vel_offsets


    def compute_ForwardKinematics(self, q):
        """
        Description:
            1. Computes the Cartesian Position of the robot using the pinnocchio libraries.
            2. The TCP position is w.r.t. the 'tip_link' of the robot.

        Args:
            q -> np.array (3,) : Joint Positions of the robot.

        Returns:
            np.array (3,): Cartesian Position of the robot.
        """

        pin.framesForwardKinematics(self.model, self.model_data, q)
        pos = np.array(self.model_data.oMf[self.EOAT_ID].translation)
        return pos


    def compute_Jacobian(self, q):
        """
        Description:
            1. Computes the Jacobian Tensor of the robot using the pinnocchio libraries.

        Args:
            q -> np.array (3,) : Joint Positions of the robot.

        Returns:
            np.array (3,3): Jacobian Tensor of the robot.
        """

        pin.computeJointJacobians(self.model, self.model_data, q)
        return pin.getFrameJacobian(self.model, self.model_data, self.EOAT_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]

    def wait(self, seconds):
        """
        Description:
            1. Makes the robot wait for few seconds

        Args:
            seconds -> Float: Seconds
        """
        sleep(seconds)

    def Move_P2P(self, goal_pos, mode='linear', C=None):
        """
        Description:
            1. Moves the robot towards the target cartesian point.
            2. The code is based of optimization. source: https://www.user.tu-berlin.de/mtoussai/teaching/Lecture-Robotics.pdf

        Args:
            goal_pos -> np.array (3,) : Target Position in Cartesian Space to move the robot.

        Keyword Args:
            1. mode -> string: Interpolation mode contains 'linear', 'joint' & 'circular'. Defaults to 'linear'.
            2. C -> list(2, (3,)): To be used when 'circular' mode is used. Estimates a 2-Point Circular Trajectory. Defaults to None.

        Returns:
            list(n, (3,)): List of the EOAT(TCP) logged positions.
        """

        # Initiatial Reading of States
        q, _ = self.offset_states()

        # Cost Metric for IK. Optimization
        W = 1e-4 * np.eye(3)

        # Track EOAT
        EOAT = []
        EOAT.append(self.compute_ForwardKinematics(q))

        def move_linear(init_y, goal_pos, smooth=0.75, max_steps=100):
            """SUB-FUNCTION to for linear interpolation

            Args:
                init_y -> np.array(3,): Current TCP Position in Cartesian Space
                goal_pos -> np.array(3,): Target Position in Cartesian Space
                max_steps -> int: No. of Intermediate Points between 'init_y' & 'goal_pos'
            """
            # Copy States
            y = init_y
            y_target = goal_pos.copy()

            while np.linalg.norm(goal_pos-y) >= 0.001:
                for i in range(1, max_steps):
                    # Read the actual position
                    q, _ = self.offset_states()
                    # Compute the Forward Kinematic Constants
                    y = self.compute_ForwardKinematics(q)
                    J = self.compute_Jacobian(q) 
                    # Compute the Singularity Robust Inverse Jacobian Tensor
                    inv_J = np.linalg.inv(J.T @ J + W) @ J.T
                    # Generate linear intermediate points in between goal pos. & start pos. 
                    y_target = init_y + (i/max_steps)*(goal_pos-y)
                    # Compute the joint angles updates using Inverse Kinematics
                    q += smooth * (inv_J @ (y_target - y) )#+ (np.eye(3) - inv_J @ J) @ (q - init_q))
                    # Move the Joints
                    self.set_JointStates(q)
                    # Log the EOAT Position
                    EOAT.append(y)
                    # Break the loop if tolerance is achieved
                    if np.linalg.norm(goal_pos-y) <= 0.001:
                        break
                # Start the loop again if tolerance is not met
                init_y = self.compute_ForwardKinematics(q)


        if mode == 'linear':
            init_y = self.compute_ForwardKinematics(q)
            move_linear(init_y, goal_pos)

        if mode == 'joint':
            max_steps = 10
            smooth = 0.75
            for i in range(1, max_steps):
                q, _ = self.offset_states()
                q += smooth * (goal_pos - q)
                self.set_JointStates(q)
                EOAT.append(self.compute_ForwardKinematics(q))
                
        if mode == 'circular':
            max_steps = 100
            #Calculate Parameters of the Circle
            P1 = C[0]
            P2 = C[1]
            P3 = C[2]
            C = (P1+P2+P3)/3

            # Compute Normal Vector to Plane
            n = np.cross((P2-P1), (P3-P1))
            n = n / np.linalg.norm(n)

            # Pick a Point on a Plane
            Q = C + np.cross(n, (P1 - C))

            theta = np.arange(0, 2*np.pi, 1/360)
            x = C[0] + np.cos(theta) * (P1[0]-C[0]) + np.sin(theta) * (Q[0]-C[0])
            y = C[1] + np.cos(theta) * (P1[1]-C[1]) + np.sin(theta) * (Q[1]-C[1])
            z = C[2] + np.cos(theta) * (P1[2]-C[2]) + np.sin(theta) * (Q[2]-C[2])

            # Traverse the Points
            for j in range(len(x)):
                q, _ = self.offset_states()
                init_y = self.compute_ForwardKinematics(q)
                y_target = np.array([x[j],y[j],z[j]])
                move_linear(init_y, y_target)
        
        return np.vstack(EOAT)


    def Trajectory_Planner(self, Point_List):
        """
        Description:
            1. Moves the robot in a path using a list of points. By default the interpolation mode is linear in between the points.

        Args:
            Point_List -> list(n,(3,)): List of cartersian space trajectory points.

        Returns:
            list(n, (3,)): List of the EOAT(TCP) logged positions.
        """

        # Track EOAT
        EOAT = []
        
        # Invoke Move_P2P for each Point
        for i, point in enumerate(Point_List):
            trace_points = self.Move_P2P(point)
            EOAT.append(trace_points)
            sleep(0.5)

        return np.vstack(EOAT)


    def plot_TracePath(self, trace_points, save=False):
        """
        Description:
            1. Generates a 3D Plot of the EOAT Position.

        Args:
            trace_points -> list(n, (3,)): List of the EOAT(TCP) logged positions.
            
        Keyword Args:
            save -> bool: Saves the 3D plot as a '.png' extension. Defaults to False.
        """

        ax = plt.axes(projection='3d')
        ax.plot3D(trace_points[:,0],trace_points[:,1],trace_points[:,2], c='red', label='Trajectory')
        ax.legend(loc='best')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        if save:
           plt.savefig('Trajectory_Plot.png') 

    ####################################################################################################################
    # JOINT DYNAMICS                                                                                      ##############
    ####################################################################################################################
    def set_JointStates(self, target_q):
        """
        Description:
            1. Moves the robot joints to designated target joint positions.

        Args:
            target_q -> np.array (3,): Target joint positions of the robot.
            
        """
        
        self.simulator.reset_state(target_q, np.zeros(3))

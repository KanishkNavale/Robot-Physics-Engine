######################################################################################################################
# DEVELOPED BY KANISHK                                                                                   #############
# MAX PLANCK INSTITUTE FOR INTELLIGENT SYSTEMS                                                           #############
# STUTTGART, GERMANY                                                                                     #############
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

    def __init__(self, comms='enp5s0', connect_hardware=True, controller=None):
        """
        Description:
            1. Initiates, Builds & Starts Robot Model components and hardware.
            2. Initializes 'NYUFingerSimulator()' as Visualizer to run at the rate of 60Hz. 
            3. Runs Inverse Dynamic Compensation as parallel process (thread).
            4. Invokes 'self.offset_states()' which reads the first starting position of robot and makes it Zero.

        Args:
            1. comms = Binds the robot communication to an ethernet port. Uses 'enp5s0' as a defualt port.
            2. connect_hardware = True (default). Connects to the robot hardware if enabled else, runs as simulation.

        Keyword Args:
            1. controller = 'None': Does not initialize any joystick mode for the robot.
                            'keyboard': Enables joystick mode on the keyboard. 
        """
        
        # Init. Hardware
        self.hardware = connect_hardware

        if self.hardware:
            try:
                self.device = NYUFingerReal()
                self.device.initialize(comms)
            except Exception as e:
                print(f'Robot Hardware Initiation Error: {e}')

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
        if self.hardware:
            q, v = self.device.get_state()
        else:
            q, v = self.simulator.get_state()

        self.joint_offsets = q
        self.vel_offsets = v

        # Initial Position
        self.pose = self.compute_ForwardKinematics(self.joint_offsets)

        # Variables for Dynamic Compensation
        q, v = self.offset_states()
        self.set_pos = q
        self.set_vel = v
        
        # Enable Dynamic Compensation
        if self.hardware:
            self.SetDynamicCompensation = Thread(target=self.DynamicCompensation)
            self.SDC_Flag = Event()
            self.SDC_Flag.clear()
            self.SetDynamicCompensation.start()

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
    # COLLISIONS COMPUTATION                                                                              ##############
    ####################################################################################################################

    def compute_Collisions(self,q):
        """
        Description:
            1. Computes the robot collision using the pinocchio library

        Args:
            q -> np.array (3,) : Joint Positions of the robot.

        Returns:
            bool: Robot in Collision
        """

        pin.computeCollisions(self.model, self.model_data, self.collision_model ,self.collision_data, q, False)

        for k in range(len(self.collision_model.collisionPairs)): 
            cr = self.collision_data.collisionResults[k]

        self.viz.displayCollisions(True)
        self.viz.displayVisuals(True)

        return cr.isCollision()


    ####################################################################################################################
    # JOINT DYNAMICS                                                                                      ##############
    ####################################################################################################################
    def set_JointStates(self, target_q, verbose=False):
        """
        Description:
            1. Moves the robot joints to designated target joint positions.
            2. invoked, disables the parallel process of Inverse Dynamic Compensation.
            3. Moves the robot to traget position using PID Controller.
            4. On reaching the target position, re-enables the Inverse Dynamic Compensation.

        Args:
            target_q -> np.array (3,): Target joint positions of the robot.
            
        Keyword Args:
            verbose: Prints interminent process information. Defaults to False.
        """
        if self.hardware:
            # Entry flag and Timer
            Entry_Flag = True
            Entry_Timer = clock()

            # Disable Dynamic Compensation
            if verbose:
                print ('Terminating Dynamic Compensation!')
            self.SDC_Flag.set()
            self.SetDynamicCompensation.join()

            # Chart for tuning
            q_log = []

            # Goal Set Points
            q_goal = target_q

            # Init. Values
            q, v = self.offset_states()

            # Joint Position Dynamics Defaults
            # Julian's P=5 D=0.1
            self.Kp = np.array([1,        3,        1.35])
            self.Kd = np.array([0.01,     0.09,     0.1])
            self.Ki = np.array([8e-4,     8e-4,     3.5e-4])

            # Error Inits.
            pre_err = np.zeros(3)
            int_err = np.zeros(3)

            for i in range(500):
                # Loop Timer
                if not Entry_Flag:
                    loop_timer = clock()

                # Compute Positional Errors
                pro_err  = q_goal - q
                der_err  = pre_err - pro_err
                int_err += pro_err
                pre_err = pro_err

                # Positional controller
                u  = self.Kp* pro_err
                u += self.Ki* int_err
                u += self.Kd* der_err

                # Send torque commands
                self.device.send_joint_torque(u)
                    
                q, v = self.offset_states()
                self.set_pos = q

                # Attempt to send joint command every 1ms
                if Entry_Flag:
                    Entry_Flag = False
                    delay = Entry_Timer - clock()
                    sleep (abs(1e-3 - delay))
                else:
                    delay = loop_timer - clock()
                    sleep (abs(1e-3 - delay))
            
            # Restart Dynamic Compensation
            if verbose:
                print ('Restarting Dynamic Compensation')
            self.SetDynamicCompensation = Thread(target=self.DynamicCompensation)
            self.SDC_Flag.clear()
            self.SetDynamicCompensation.start()
        
        else:
            self.simulator.reset_state(target_q, np.zeros(3))


    ####################################################################################################################
    # GRAVITY COMPENSATION + Other Inverse Dynamics Compensations                                         ##############
    ####################################################################################################################

    def DynamicCompensation(self, verbose=False):
        """
        Description:
            1. Managed by a thread process.
            2. Helps to retain the position in the real world against external forces.
            3. Command cycle rate is 1000Hz

        Keyword Args:
            verbose: Prints interminent process information. Defaults to False.
        """

        if verbose:
            print ('Setting in Dynamic Compensation!')

        while not self.SDC_Flag.is_set():
            # Computing Inverse Dynamics for hold torque
            timer = clock()

            #gu = pin.computeGeneralizedGravity(self.model, self.model_data, self.set_pos)
            tau = pin.rnea(self.model, self.model_data, self.set_pos, self.set_vel, np.zeros(3))
            self.device.send_joint_torque(tau)

            # Attempt to send an joint command every 1ms
            compute_time = clock() - timer
            sleep(abs(1e-3 - compute_time))      
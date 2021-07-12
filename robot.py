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
from forge.robot_simulator import NYUFingerSimulator

# Using OpenRobotics Scripts
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

    def __init__(self, device, comms):
        # Init. Hardware
        try:
            self.device = device
            self.device.initialize(comms)
        except Exception as e:
            print(f'Robot Hardware Initiation Error: {e}')

        # Init. Simulation Env. using .urdf file
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
        """ Using the Simulator to display the model """
        self.simulator = NYUFingerSimulator(self.package_dir, self.urdf_path)
            
        # Get the FrameID of the EOAT
        self.EOAT_ID = self.model.getFrameId('finger_tip_link')
        # Get the FrameID of the Base
        self.Base_ID = self.model.getFrameId('base_link')

        # Offset Joint Values
        q, v = self.device.get_state()
        self.joint_offsets = q
        self.vel_offsets = v

        # Initial Position
        self.pose = self.compute_ForwardKinematics(self.joint_offsets)

        # Variables for Dynamic Compensation
        q, v = self.calibrated_states()
        self.set_pos = q
        self.set_vel = v
        
        # Enable Dynamic Compensation
        self.SetDynamicCompensation = Thread(target=self.DynamicCompensation)
        self.SDC_Flag = Event()
        self.SDC_Flag.clear()
        self.SetDynamicCompensation.start()

        # Refresh Display
        self.update_simulator = Thread(target=self.refresh_viz)
        self.update_simulator.start()

        # Use Keyboard as a controller
        pygame.init()
        pygame.display.set_mode((300, 300))
        #self.controller = Thread(target=self.keyboard_listener)
        #self.controller.start()


    ####################################################################################################################
    # Refresh Display                                                                                     ##############
    ####################################################################################################################
    def refresh_viz(self):
        refresh_rate = 60 #(60FPS)
        while True:
            q, v = self.calibrated_states()
            self.simulator.reset_state(q, v)
            sleep(1/refresh_rate)


    ####################################################################################################################
    # Keyboard Listener                                                                                   ##############
    ####################################################################################################################
    def keyboard_listener(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_KP6:
                        print (f'Moving Robot: Direction +Y')
                        self.pose = self.pose + np.array([0, 0.005, 0])
                        self.P2P_Planner(self.pose)
                    if event.key == pygame.K_KP4:
                        print (f'Moving Robot: Direction -Y')
                        self.pose = self.pose + np.array([0, -0.005, 0])
                        self.P2P_Planner(self.pose)
                    if event.key == pygame.K_KP8:
                        print (f'Moving Robot: Direction -X')
                        self.pose = self.pose + np.array([-0.005, 0.0, 0])
                        self.P2P_Planner(self.pose)
                    if event.key == pygame.K_KP2:
                        print (f'Moving Robot: Direction +X')
                        self.pose = self.pose + np.array([0.005, 0.0, 0])
                        self.P2P_Planner(self.pose)
                    if event.key == pygame.K_UP:
                        print (f'Moving Robot: Direction +Z')
                        self.pose = self.pose + np.array([0.0, 0.0, 0.005])
                        self.P2P_Planner(self.pose)
                    if event.key == pygame.K_DOWN:
                        print (f'Moving Robot: Direction -Z')
                        self.pose = self.pose + np.array([0.0, 0.0, -0.005])
                        self.P2P_Planner(self.pose)


    ####################################################################################################################
    # KINEMATICS                                                                                          ##############
    ####################################################################################################################
    # Compute Joint Offsets
    def calibrated_states(self):
        q, v = self.device.get_state()
        return q - self.joint_offsets, v - self.vel_offsets


    # Computing Forward Kinematics
    def compute_ForwardKinematics(self, q):
        pin.framesForwardKinematics(self.model, self.model_data, q)
        pos = np.array(self.model_data.oMf[self.EOAT_ID].translation)
        return pos


    # Compute the Jacobian
    def compute_Jacobian(self, q):
        pin.computeJointJacobians(self.model, self.model_data, q)
        return pin.getFrameJacobian(self.model, self.model_data, self.EOAT_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]


    # Trajectory Planning for Point-to-Point(P2P) Motion
    def P2P_Planner(self, goal_pos, interpolation='linear'):
        # Initiate Callers
        q, v = self.calibrated_states()
        y_target = goal_pos.copy()
        init_y = self.compute_ForwardKinematics(q)

        # Constants
        W = 1e-4 * np.eye(3)

        # Track EOAT
        EOAT = []
        EOAT.append(init_y)

        # HyperParams-> Needs to be tuned for large P2P Motion
        if interpolation=='joint':
            max_steps = 10
            smooth = 0.5
        if interpolation=='linear':
            max_steps = 10
            smooth = 0.75

        for i in range(1, max_steps):
            y = self.compute_ForwardKinematics(q)
            J = self.compute_Jacobian(q) 
            if interpolation=='joint':
                pass
            if interpolation=='linear':
                y_target = init_y + (i/max_steps)*(goal_pos-y)

            # Compute Step Angle
            q += smooth * np.linalg.inv(J.T @ J + W) @ J.T @ (y_target - y)
            self.set_JointStates(q)

            EOAT.append(y)

        return np.vstack(EOAT)
    

    # Point based Trajectory Planner
    def Trajectory_Planner(self, Point_List):
        # Track EOAT
        EOAT = []
        # Invoke P2P_Planner for each Point
        for i, point in enumerate(Point_List):
            trace_points = self.P2P_Planner(point)
            EOAT.append(trace_points)

        return np.vstack(EOAT)


    # Plot of EOAT Position 
    def plot_EOATPosition(self, trace_points, save=False):
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
    # Definition to Compute Collisions
    def compute_Collisions(self,q):
        pin.computeCollisions(self.model, self.model_data, self.collision_model ,self.collision_data, q, False)
        for k in range(len(self.collision_model.collisionPairs)): 
            cr = self.collision_data.collisionResults[k]
        self.viz.displayCollisions(True)
        self.viz.displayVisuals(True)
        return cr.isCollision()


    ####################################################################################################################
    # JOINT DYNAMICS                                                                                      ##############
    ####################################################################################################################
    # Definition to Set Joint Torques using dynamics
    def set_JointStates(self, target_q):

        # Entry flag and Timer
        Entry_Flag = True
        Entry_Timer = clock()

        # Disable Dynamic Compensation
        print ('Terminating Dynamic Compensation!')
        self.SDC_Flag.set()
        self.SetDynamicCompensation.join()

        # Chart for tuning
        q_log = []

        # Goal Set Points
        q_goal = target_q

        # Init. Values
        q, v = self.calibrated_states()

        
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

            q, v = self.calibrated_states()
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
        print ('Restarting Dynamic Compensation')
        self.SetDynamicCompensation = Thread(target=self.DynamicCompensation)
        self.SDC_Flag.clear()
        self.SetDynamicCompensation.start()


    ####################################################################################################################
    # GRAVITY COMPENSATION + Other Inverse Dynamics Compensations                                         ##############
    ####################################################################################################################
    # Definition to compute Inverse Dynamics Gravity + Inertia + NLE Forces + Centrifugal & Corrolis 
    def DynamicCompensation(self):
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
        
            
  

#######################################################################################################################
# MAIN Runtime                                                                                           ##############
#######################################################################################################################
if __name__ == "__main__":
    robot  = Robot(NYUFingerReal(), 'enp5s0')
    robot.set_JointStates(np.ones(3))
        
    
    


        

######################################################################################################################
# DEVELOPED BY KANISHK                                                                                   #############
# MAX PLANCK INSTITUTE FOR INTELLIGENT SYSTEMS                                                           #############
# STUTTGART, GERMANY                                                                                     #############
                                                                                                         #############
# THIS SCRIPT CONSTRUCTS A ROBOT CLASS BASED ON FUNCTIONALITIES FROM THE 'FORGE'                         #############
######################################################################################################################

# Library Imports
from numpy.lib.npyio import save
from nyu_finger import NYUFingerReal

import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

import numpy as np
np.set_printoptions(precision=4)

import matplotlib.pyplot as plt
plt.style.use('dark_background')

import os
from threading import Thread
from time import sleep



#######################################################################################################################
# Class Construct of the robot                                                                            #############
#######################################################################################################################
class Robot:
    """ Contructs a class for the 'nyu_finger' robot """

    def __init__(self, device, comms):
        # Init. Hardware
        self.device = device
        try:
            self.device.initialize(comms)
        except:
            print('Something Went Wrong!')

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
        self.viz = GepettoVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer()
        self.viz.loadViewerModel("nyu_finger")
        self.viz.displayCollisions(True)
        self.viz.displayVisuals(True)
        q = pin.neutral(self.model)
        q, _ = self.device.get_state()
        self.viz.display(q)

        # Get the FrameID of the EOAT
        self.EOAT_ID = self.model.getFrameId('finger_tip_link')
        # Get the FrameID of the Base
        self.Base_ID = self.model.getFrameId('base_link')
        
        # Enable Dynamic Compensation
        q, v = self.device.get_state()
        self.SetDynamicCompensation = Thread(target=self.set_DynamicCompensation, args=(True, q, v, np.zeros(3)))
        self.SetDynamicCompensation.start()

    ####################################################################################################################
    # KINEMATICS                                                                                          ##############
    ####################################################################################################################
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
    def P2P_Planner(self, init_pos, goal_pos, interpolation='joint', track_EOAT=True):
        # Initiate Callers
        q = init_pos
        y_target = goal_pos.copy()
        init_y = self.compute_ForwardKinematics(q)

        # Constants
        W = 1e-4 * np.eye(3)

        # Track EOAT
        if track_EOAT:
            EOAT = []
            EOAT.append(init_y)

        # HyperParams-> Needs to be tuned for large P2P Motion
        if interpolation=='joint':
            max_steps = 100
            smooth = 0.1
        if interpolation=='linear':
            max_steps = 1000
            smooth = 0.5

        for i in range(1, max_steps):
            y = self.compute_ForwardKinematics(q)
            J = self.compute_Jacobian(q) 
            if interpolation=='joint':
                pass
            if interpolation=='linear':
                y_target = init_y + (i/max_steps)*(goal_pos-y)

            # Compute Step Angle
            q += smooth * np.linalg.inv(J.T @ J + W) @ J.T @ (y_target - y)
            #self.set_JointStates(q)

            self.viz.display(q)
            sleep(2e-3)

            if track_EOAT:
                EOAT.append(y)

        if track_EOAT:
            return q, np.vstack(EOAT)
        else:
            return q
    

    # Point based Trajectory Planner
    def Trajectory_Planner(self, joint_pos, Point_List, track_EOAT=True):
        q = joint_pos
        # Track EOAT
        if track_EOAT:
            EOAT = []
        # Invoke P2P_Planner for each Point
        for i, point in enumerate(Point_List):
            q, trace_points = self.P2P_Planner(q, point)
            EOAT.append(trace_points)
        if track_EOAT:
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
        # Disable Dynamic Compensation
        self.SetDynamicCompensation.join()
        
        tau = .01
        N = 5
        T = N*tau

        # Goal Set Points
        q_goal = target_q
        v_goal = np.zeros(3)
        a_goal = np.zeros(3)

        # Init. Values
        q, v = self.device.get_state()

        # Joint Dynamics Defaults
        self.Kp = 1.5
        self.Kd = 0.15
        self.Ki = 1e-4

        # For the Ki Tuning
        Err_log=[]
        Err_log.append(q_goal - q)

        print ('XYZ')
        
        for i in range(1, N+1):
            t = i * tau
            
            # Compute Step Angles
            q = q_goal - (q*t)/T
            v = v_goal - (v*t)/T
            a = a_goal - (a*t)/T

            # PID controller
            u = self.Kp*(q_goal - q) + self.Ki* np.sum(Err_log) + self.Kd*(v_goal - v) 

            # Send torque commands
            self.device.send_joint_torque(u)
            print (q_goal, q)

            q, v = self.device.get_state()
            self.viz.display(q)

            Err_log.append(q_goal - q)
            sleep(tau)
            
        # Restart Dynamic Compensation
        self.SetDynamicCompensation.start()


    ####################################################################################################################
    # GRAVITY COMPENSATION + Other Inverse Dynamics Compensations                                          ##############
    ####################################################################################################################
    # Definition to compute Inverse Dynamics Gravity + Inertia + NLE Forces + Centrifugal & Corrolis 
    def set_DynamicCompensation(self, enable, set_state, set_velocity, set_acc):
        while(enable):
            # Computing Inverse Dynamics for stall torque
            tau = pin.rnea(self.model, self.model_data, set_state, set_velocity, set_acc)
            self.device.send_joint_torque(tau)
  

    
#######################################################################################################################
# MAIN Runtime                                                                                           ##############
#######################################################################################################################
if __name__ == "__main__":
    robot = Robot(NYUFingerReal(), 'enp5s0')
    print ('Kanishk')
    robot.set_JointStates(np.zeros(3))
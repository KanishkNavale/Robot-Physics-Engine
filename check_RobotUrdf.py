# Library Imports
import numpy as np
from nyu_finger import NYUFingerReal
import pinocchio as pin
from time import sleep

# Class of the Robot
class Robot:
    def __init__(self, device, comms):
        # Init. Hardware
        self.device = device
        self.device.initialize(comms)

        # Init. Simulation Env. using .urdf file
        self.urdf_file = '../urdf/fingeredu.urdf'
        self.model = pin.buildModelFromUrdf(self.urdf_file)
        self.data = self.model.createData()

        # Get the EOAT frame ID
        self.EOATFrameID = 12
        print (self.EOATFrameID)

    def print_RobotConfiguration(self):
        """ Prints the Kinematic Attributes of the Robot  """
        print (f'Robot Joints: {self.model}')
        print (f'Robot Config. Space Dim.: {self.model.nq}')
        print (f'Robot DOF: {self.model.nv}')
        print (f'Robot Joint Count: {self.model.njoints}')
        print (f'Robot Frames:')
        for i,f in enumerate(self.model.frames): print(i,f.name,f.parent)

    def read_ActuatorStates(self):
        """ Returns the Current Position & Velocity of the Robot """
        pos, vel = self.device.get_state()
        return pos, vel

    def set_ActutatorStates(self, tau):
        """ Sets the Actuators based on tau = Derived from Inverse Dynamics """
        self.device.send_joint_torque(tau)

    def compute_mass(self, units= 'kg'):
        """ Computes and Return the Mass of the Robots """
        q, _ = self.read_ActuatorStates()
        M = pin.crba(self.model, self.data,q)
        if units == 'kg':
            return (np.linalg.eig(M)[0])
        if units == 'grams':
            return (np.linalg.eig(M)[0]*1e3)

    def compute_ForwardKin(self, q=None):
        """ Returns the Computed Forward Kinematics of the Robot w.r.t EOAT """
        if q is None:
            q,_ = self.read_ActuatorStates()
        
        # Compute the EOAT Position
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model,self.data)

        return self.data.oMf[self.EOATFrameID].translation.T

    def compute_Jacobian(self, q=None):
        """ Computes & Returns the Positional Jacobian """
        if q is None:
            q, v = self.read_ActuatorStates()

        LOCAL = pin.ReferenceFrame.LOCAL

        pin.forwardKinematics(self.model,self.data,q)
        pin.computeJointJacobians(self.model,self.data,q)
        pin.updateFramePlacements(self.model,self.data)
        return pin.getFrameJacobian(self.model, self.data, self.EOATFrameID, LOCAL)[:3]

    def gen_Trajectory(self, target=np.array([0,0,0]), steps=10, smooth=True, verbose=False):
        """ Moves the robot to target position """
        # Traget Initialization
        y_target = target

        # Initialization Part
        n = self.model.nq
        W = 1e-4 * np.identity(n)
        q, v = self.read_ActuatorStates()
        
        # Smooth Trajectory Generation 
        # Smooth = HYPER PARAMS.
        if smooth:
            smooth = 1e-2
        else:
            smooth = 1

        for i in range(steps):
            #q,v = self.read_ActuatorStates()
            y = self.compute_ForwardKin(q=q)
            J = self.compute_Jacobian(q=q)

            
            q += smooth * np.linalg.inv(J.T @ J + W) @ J.T @ (y_target - y)     
            
            print (np.linalg.norm(y_target - y), y_target, y, q)
            sleep(1)
            
                    
# Main    
if __name__ == "__main__":
    robot = Robot(NYUFingerReal(), 'enp5s0')
    
    robot.gen_Trajectory(smooth=False)
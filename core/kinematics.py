import numpy as np
import pinocchio as pin


def ForwardKinematics(self, q: np.zeros(3), v: np.zeros(3)) -> np.ndarray:

    index = self.EOAT_ID
    frame = pin.ReferenceFrame.LOCAL
    pin.forwardKinematics(self.model, self.model_data, q, v)
    pin.updateFramePlacements(self.model, self.model_data)
    pos = pin.updateFramePlacement(self.model, self.model_data, index)
    vel = pin.getFrameVelocity(self.model, self.model_data, index, frame)
    return np.array(pos)[:3, -1], np.array(vel)[:3]


def FrameJacobian(self, q: np.ndarray) -> np.ndarray:

    frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    pin.computeJointJacobians(self.model, self.model_data, q)
    return pin.getFrameJacobian(self.model, self.model_data, self.EOAT_ID, frame)[:3]


def FrameHessian(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:

    frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    pin.computeJointJacobians(self.model, self.model_data, q)
    pin.framesForwardKinematics(self.model, self.model_data, q)
    pin.computeJointJacobiansTimeVariation(self.model, self.model_data, q, v)
    return pin.getJointJacobianTimeVariation(self.model, self.model_data, self.EOAT_ID, frame)[:3]

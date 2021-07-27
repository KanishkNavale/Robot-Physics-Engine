import os.path
import numpy as np

from robot_properties_nyu_finger.config import NYUFingerConfig
from nyu_finger.nyu_finger_hwp_cpp import NYUFingerHWP

if __name__ == "__main__":
    finger = NYUFingerHWP()

    finger.initialize(os.path.join(
        NYUFingerConfig.dgm_yaml_dir, 'dgm_parameters_nyu_finger.yaml'))

    finger.run()

    print()
    input("Press enter to start calibration.")

    finger.calibrate(np.zeros(3))
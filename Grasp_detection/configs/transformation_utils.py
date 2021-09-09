import numpy as np
import transforms3d


def global_to_local_transformation(frame, point):
    T_global_to_local = np.eye(4)
    T_global_to_local[0:3, 0:3] = frame.T
    T_global_to_local[0:3, 3] = -np.dot(frame.T, point)
    return T_global_to_local


def local_to_global_transformation_quat(quat, point):
    T_local_to_global = np.eye(4)
    frame = transforms3d.quaternions.quat2mat(quat)
    T_local_to_global[0:3, 0:3] = frame
    T_local_to_global[0:3, 3] = point

    return T_local_to_global

import math
import torch
import numpy as np


def _get_local_search_to_local(LOCAL_TO_LOCAL_SEARCH):
    NUMPY_LOCAL_TO_LOCAL_SEARCH = LOCAL_TO_LOCAL_SEARCH.cpu().numpy()
    NUMPY_LOCAL_SEARCH_TO_LOCAL = np.zeros([LOCAL_TO_LOCAL_SEARCH.shape[0], 4, 4])
    for j in range(LOCAL_TO_LOCAL_SEARCH.shape[0]):
        NUMPY_LOCAL_SEARCH_TO_LOCAL[j, :, :] = np.linalg.inv(NUMPY_LOCAL_TO_LOCAL_SEARCH[j, :, :])
    return NUMPY_LOCAL_SEARCH_TO_LOCAL


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# torch.cuda.set_device(2)
# Point Cloud Pre-processing
TABLE_HEIGHT = 0.5   #0.75  0.5
SAMPLE_REGION = TABLE_HEIGHT + 0.015
# Workspace should be (6,): low_x, high_x, low_y, high_y, low_z, high_z
WORKSPACE = [-0.50, 0.50, -0.50, 0.50, TABLE_HEIGHT-0.04, TABLE_HEIGHT + 0.45]  #-0.04
WORKSPACE_SCENE = [-1.00, 1.50, -1.5, 1.2, TABLE_HEIGHT - 0.04, TABLE_HEIGHT + 0.45]
VOXEL_SIZE = 0.005
NUM_POINTS_THRESHOLD = 8
RADIUS_THRESHOLD = 0.04

# Scene Point Cloud
SCENE_MULTIPLE = 8  # The density of points in scene over view cloud

# Normal Estimation
NORMAL_RADIUS = 0.01
NORMAL_MAX_NN = 30

# Local Frame Search
LENGTH_SEARCH = [-0.06, -0.04, -0.02, -0.00]
#LENGTH_SEARCH = [-0.040, -0.030, -0.015, -0.000]
THICKNESS_SEARCH = [0]
THETA_SEARCH = list(range(-90, 90, 20)) # 15
CURVATURE_RADIUS = 0.01
BACK_COLLISION_THRESHOLD = 0 * math.sqrt(
    SCENE_MULTIPLE)  # if more than this number of points exist behind the back of hand, grasp fail
BACK_COLLISION_MARGIN = 0.0  # points that collide with back hand within this range will not be detected
FINGER_COLLISION_THRESHOLD = 0
CLOSE_REGION_MIN_POINTS = 10
for i in range(len(THETA_SEARCH)):
    THETA_SEARCH[i] /= 57.29578

# Antipodal Grasp
NEIGHBOR_DEPTH = torch.tensor(0.005).to(device)

# Gripper Configuration
#'''
HALF_BOTTOM_WIDTH = 0.12/2 + 0.01  #0.06
BOTTOM_LENGTH = 0.12
FINGER_WIDTH = 0.01
HALF_HAND_THICKNESS = 0.005
FINGER_LENGTH = 0.06
HAND_LENGTH = BOTTOM_LENGTH + FINGER_LENGTH
HALF_BOTTOM_SPACE = HALF_BOTTOM_WIDTH - FINGER_WIDTH
'''
HALF_BOTTOM_WIDTH = 0.057
BOTTOM_LENGTH = 0.08
FINGER_WIDTH = 0.023
HALF_HAND_THICKNESS = 0.012
FINGER_LENGTH = 0.09
HAND_LENGTH = BOTTOM_LENGTH + FINGER_LENGTH
HALF_BOTTOM_SPACE = HALF_BOTTOM_WIDTH - FINGER_WIDTH
'''

GRIPPER_BOUND = np.ones([4, 8])
i = 0
for x in [FINGER_LENGTH, -BOTTOM_LENGTH]:
    for y in [HALF_BOTTOM_WIDTH, -HALF_BOTTOM_WIDTH]:
        for z in [HALF_HAND_THICKNESS, -HALF_HAND_THICKNESS]:
            GRIPPER_BOUND[0:3, i] = [x, y, z]
            i += 1
TORCH_GRIPPER_BOUND = torch.tensor(GRIPPER_BOUND, device=device).float()

INDEX_TO_ARRAY = []
GRASP_PER_LENGTH = len(THETA_SEARCH) * len(THICKNESS_SEARCH)
for length in LENGTH_SEARCH:
    for theta in THETA_SEARCH:
        for height in THICKNESS_SEARCH:
            INDEX_TO_ARRAY.append((length, theta, height))

_INDEX_TO_ARRAY_TORCH = torch.tensor(INDEX_TO_ARRAY).to(device)
LOCAL_TO_LOCAL_SEARCH = torch.eye(4).unsqueeze(0).expand(size=(len(INDEX_TO_ARRAY), 4, 4))
LOCAL_TO_LOCAL_SEARCH = LOCAL_TO_LOCAL_SEARCH.contiguous()
LOCAL_TO_LOCAL_SEARCH[:, 0, 3] = -_INDEX_TO_ARRAY_TORCH[:, 0]
LOCAL_TO_LOCAL_SEARCH[:, 2, 3] = -_INDEX_TO_ARRAY_TORCH[:, 2]
LOCAL_TO_LOCAL_SEARCH[:, 1, 1] = torch.cos(_INDEX_TO_ARRAY_TORCH[:, 1])
LOCAL_TO_LOCAL_SEARCH[:, 2, 2] = torch.cos(_INDEX_TO_ARRAY_TORCH[:, 1])
LOCAL_TO_LOCAL_SEARCH[:, 1, 2] = torch.sin(_INDEX_TO_ARRAY_TORCH[:, 1])
LOCAL_TO_LOCAL_SEARCH[:, 2, 1] = -torch.sin(_INDEX_TO_ARRAY_TORCH[:, 1])
LOCAL_TO_LOCAL_SEARCH = LOCAL_TO_LOCAL_SEARCH.to(device)
NUMPY_LOCAL_SEARCH_TO_LOCAL = _get_local_search_to_local(LOCAL_TO_LOCAL_SEARCH)
TORCH_LOCAL_SEARCH_TO_LOCAL = torch.inverse(LOCAL_TO_LOCAL_SEARCH)

# Table collision check
LOCAL_SEARCH_TO_LOCAL = torch.inverse(LOCAL_TO_LOCAL_SEARCH).contiguous()
TABLE_COLLISION_OFFSET = 0.005

# GPD Projection Configuration
GRASP_NUM = 600
PROJECTION_RESOLUTION = 60
PROJECTION_MARGIN = 1

CAMERA_POSE = [
    [0.8, 0, 2.1, 0.6912141595435636, 0.14907388569241384, 0.149073900960075, 0.6912141016382668],
    [0.8, -0.8, 2.1, 0.8867226612461571, 0.2593767218260428, 0.1074373143136802, 0.36729261046251954],
    [0.8, 0.8, 2.1, 0.36729255283959544, 0.10743736746243611, 0.25937673077155354, 0.886722676058065],
    [-0.8, 0, 2.1, 0.6912141595435636, 0.14907388569241384, -0.149073900960075, -0.6912141016382668],
    [-0.8, 0.8, 2.1, 0.36729255283959544, 0.10743736746243611, -0.25937673077155354, -0.886722676058065],
    [-0.8, -0.8, 2.1, 0.8867226612461571, 0.2593767218260428, -0.1074373143136802, -0.36729261046251954],
    [0, -0.8, 2.1, 0.977524397985601, 0.21082232173773435, 0.0, 2.3269508847313138e-17],
    [0, 0, 2.1, 1.0, 0.0, 0.0, 0.0],
    [0, 0.8, 2.1, 7.75993049707176e-08, -1.694861304076286e-09, 0.21082230717148598, 0.9775244011270945]
]

'''
CAMERA_POSE = [
    [0.8, 0, 1.7, 0.948, 0, 0.317, 0],
    [-0.8, 0, 1.6, -0.94, 0, 0.342, 0],
    [0.0, 0.75, 1.7, 0.671, -0.224, 0.224, 0.671],
    [0.0, -0.75, 1.6, -0.658, -0.259, -0.259, 0.658]
]
'''

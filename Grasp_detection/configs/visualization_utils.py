import open3d
import numpy as np
from configs import config
from configs.path import get_resource_dir_path
import socket
import json
import cv2
import pdb

hostname = socket.gethostname()
scene_dir = get_resource_dir_path('scene')

#create_box = open3d.geometry.TriangleMesh.create_box
create_box = open3d.geometry.create_mesh_box

def get_hand_geometry(T_global_to_local, color=(0.1, 0.6, 0.3)):
    back_hand = create_box(height=2 * config.HALF_BOTTOM_WIDTH,
                           depth=config.HALF_HAND_THICKNESS * 2,
                           width=config.BOTTOM_LENGTH - config.BACK_COLLISION_MARGIN)
    # back_hand = open3d.geometry.TriangleMesh.create_cylinder(height=0.1, radius=0.02)

    temp_trans = np.eye(4)
    temp_trans[0, 3] = -config.BOTTOM_LENGTH
    temp_trans[1, 3] = -config.HALF_BOTTOM_WIDTH
    temp_trans[2, 3] = -config.HALF_HAND_THICKNESS
    back_hand.transform(temp_trans)

    finger = create_box((config.FINGER_LENGTH + config.BACK_COLLISION_MARGIN),
                        config.FINGER_WIDTH,
                        config.HALF_HAND_THICKNESS * 2)
    finger.paint_uniform_color(color)
    back_hand.paint_uniform_color(color)
    #left_finger = copy.deepcopy(finger)
    left_finger = create_box((config.FINGER_LENGTH + config.BACK_COLLISION_MARGIN),
                        config.FINGER_WIDTH,
                        config.HALF_HAND_THICKNESS * 2)
    left_finger.paint_uniform_color(color)

    temp_trans = np.eye(4)
    temp_trans[1, 3] = config.HALF_BOTTOM_SPACE
    temp_trans[2, 3] = -config.HALF_HAND_THICKNESS
    temp_trans[0, 3] = -config.BACK_COLLISION_MARGIN
    left_finger.transform(temp_trans)
    temp_trans[1, 3] = -config.HALF_BOTTOM_WIDTH
    finger.transform(temp_trans)

    # Global transformation
    T_local_to_global = np.linalg.inv(T_global_to_local)

    back_hand.transform(T_local_to_global)
    finger.transform(T_local_to_global)
    left_finger.transform(T_local_to_global)

    vis_list = [back_hand, left_finger, finger]
    for vis in vis_list:
        vis.compute_vertex_normals()
    return vis_list


def create_point_sphere(pos):
    sphere = open3d.geometry.create_mesh_sphere(0.08)
    trans = np.eye(4)
    trans[0:3, 3] = pos
    sphere.transform(trans)
    sphere.paint_uniform_color([0, 1, 0])
    return sphere


def create_coordinate_marker(frame, point):
    p = [point, point + frame[:, 0], point + frame[:, 1], point + frame[:, 2]]
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.Vector3dVector(p)
    line_set.lines = open3d.Vector2iVector(lines)
    line_set.colors = open3d.Vector3dVector(colors)
    return line_set


def create_local_points_with_normals(points, normals):
    """
    Draw points and corresponding normals in local frame with gripper in (0, 0, 0)
    :param points: (3, n) np.array
    :param normals: (3, n) np.array
    :return: Open3d geometry list
    """
    hand = get_hand_geometry(np.eye(4))
    assert points.shape == normals.shape
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.Vector3dVector(points.T)
    pc.normals = open3d.Vector3dVector(normals.T)
    return [pc] + hand

def get_hand_rectangle(hand, camera_exmatrix, camera_matrix):
    '''
    camera_path = '/home/best/data/2/camera_info.json'
    with open(camera_path, 'r') as f:
        data = json.load(f)
        camera_exmatrix = np.array(data['extrinsic'])
        camera_matrix = np.array(data['intrinisic']).reshape((3,3))
    '''
    left_finger = hand[1]
    #open3d.visualization.draw_geometries([left_finger])
    right_finger = hand[2]
    left_points = np.array(left_finger.vertices)
    right_points = np.array(right_finger.vertices)
    cube = []

    '''
    cube_d = []
    for i in range(left_points.shape[0]):
        for j in range(right_points.shape[0]):
            vec = left_points[i] - right_points[j]
            if round(np.sqrt(np.sum(np.square(vec))), 5) == config.HALF_BOTTOM_SPACE * 2:
                #print(i,j)
                cube_d.append(left_points[i])
                cube_d.append(right_points[j])
    '''

    cube.append(left_points[2])
    cube.append(right_points[6])
    cube.append(left_points[3])
    cube.append(right_points[7])
    R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    cube = np.array(cube)
    cube = cube[np.lexsort(cube.T)][0:4]
    cube_3 = np.row_stack((cube.T, np.ones(cube.T.shape[1])))

    camera_local = np.dot(np.linalg.inv(camera_exmatrix), cube_3)
    middle_local = np.dot(R, camera_local[0:3, :])
    pixel_local = np.dot(camera_matrix, middle_local[0:3, :])  #.reshape((3,4))

    for i in range(pixel_local.shape[1]):
        pixel_local[:,i] /= (pixel_local[2,i])

    return pixel_local


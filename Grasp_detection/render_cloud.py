import bpy
import numpy as np
import os
import sys
import blensor
#import open3d
#import transforms3d
import time
import json
from mathutils import Matrix, Vector, Euler, Quaternion
#from pyquaternion import Quaternion
import pdb

# Blender python internally do not find modules in the current workspace, we need to add it explicitly
sys.path.append(os.getcwd())
from configs.dataset_config import read_state_from_json
from configs.path import get_resource_dir_path

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


# camera_to_world = [np.array([[ 7.54978995e-08, -9.11107897e-01,  4.12167926e-01,  8.00000012e-01],
#                             [ 1.00000000e+00,  8.46019382e-08,  3.84210991e-09,  0.00000000e+00],
#                             [-3.83707821e-08,  4.12167926e-01,  9.11107897e-01,  2.09999990e+00],
#                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
#                    np.array([[ 0.76620386,  0.        , -0.64259757, -0.8       ],
#                              [ 0.        ,  1.        ,  0.        ,  0.        ],
#                              [ 0.64259757,  0.        ,  0.76620386,  1.6       ],
#                              [ 0.        ,  0.        ,  0.        ,  1.        ]]),
#                    np.array([[ 1.11022302e-16, -1.00000000e+00, -5.55111512e-17, 0.00000000e+00],
#                              [ 7.99463248e-01,  1.11022302e-16,  6.00715004e-01, 7.50000000e-01],
#                              [-6.00715004e-01,  5.55111512e-17,  7.99463248e-01, 1.70000000e+00],
#                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
#                    np.array([[ 0.        ,  1.        ,  0.        ,  0.        ],
#                              [-0.73170015,  0.        , -0.68162665, -0.75      ],
#                              [-0.68162665,  0.        ,  0.73170015,  1.6       ],
#                              [ 0.        ,  0.        ,  0.        ,  1.        ]])
#                 ]

# Convention: pos + quaternion, where quaternion is [w,x,y,z]
# dataset_path = '/data/cxg8/cleaned_3DVMRD'
# data_path = os.path.join(dataset_path, 'train')
# SAVE_DIR = get_resource_dir_path('rendered_simple')
# if not os.path.exists(SAVE_DIR):
#     os.mkdir(SAVE_DIR)

dataset_path = '/data1/cxg7/models/file'
state_file_path = '/data1/cxg7/models/file'
# data_test = os.path.join(dataset_path, 'test')
# data_seen_val = os.path.join(dataset_path, 'seen_val')
data_seen_test = os.path.join(dataset_path, 'seen_test')
data_seenval = os.path.join(dataset_path, 'seenval_supermini')
data_unseen = os.path.join(dataset_path, 'unseen_val')
data_file_path = [data_unseen]


cam_intrisnic = [1791.3843994140625, 0.0, 640.0,\
                         0.0,1791.3843994140625, 480.0, \
                         0.0, 0.0, 1.0]

# def import_objects(name_list, scene):
#         for i, name in enumerate(name_list):
#             find = False
#             obj_path = os.path.join(self.obj_dir, scene, name + '.obj')
#             bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='X', axis_up='Z')
#             all_name = list(bpy.data.objects.keys())
#             not_allocated_name = [n for n in all_name if n not in list(self.name_mapping.values())]
#             for bpy_name in not_allocated_name:
#                 if bpy_name.split('_')[0] == name.split('-')[-1]:
#                     obj = bpy.data.objects[bpy_name]
#                     obj.location[2] = -10
#                     obj.hide_render = True
#                     find = True
#                     self.name_mapping.update({name: bpy_name})
#                     break
#             assert find, 'Do not find objects in blender'

class BlensorSceneServer:
    def __init__(self, preload=False):
        # self.obj_dir = get_resource_dir_path('obj')
        # self.name_list = name_list
        # self.scene = scene
        self.name_mapping = {}
        self.scanner = bpy.data.objects["Camera"]
        # self.import_objects(self.name_list, self.scene
        # self.scanner.kinect_xres = 1280
        # self.scanner.kinect_yres = 960
        # self.scanner.kinect_flength = 1791.3843994140625 * 0.0078  #13.72
        # self.scanner.data.shift_x = -(1791.3843994140625-1280/2.0) / 1280   #-0.900
        # self.scanner.data.shift_y = (1791.3843994140625-960/2.0) / 1280    #1.025
        # bpy.context.scene.render.resolution_x = 1280
        # bpy.context.scene.render.resolution_y = 960

        self.scanner.rotation_mode = 'QUATERNION'

    def import_objects(self, pathdir, name_list, scene):
        de = list(bpy.data.objects.keys())
        bpy.ops.object.select_all(action='DESELECT')
        for obj in de:
            print(obj)
            if obj == 'Camera' or obj == 'Lamp' or obj== 'Plane':
                continue
            else:
                bpy.data.objects[obj].select = True
                # bpy.context.scene.objects.active = bpy.data.objects[obj]
                bpy.ops.object.delete()
        name_mapping = {}
        for i, name in enumerate(name_list):
            print(i)
            find = False
            obj_path = os.path.join(pathdir, 'obj', scene, name + '.obj')
            bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='X', axis_up='Z')
            all_name = list(bpy.data.objects.keys())
            # not_allocated_name = [n for n in all_name if n not in list(self.name_mapping.values())]
            # for bpy_name in not_allocated_name:
            # if all_name[0].split('_')[0] == name.split('-')[-1]:
            #     not_allocated_name.append(all_name[0])
            for bpy_name in all_name:
                if bpy_name.split('_')[0] == name.split('-')[-1]:
                    obj = bpy.data.objects[bpy_name]
                    obj.location[2] = -10
                    obj.hide_render = True
                    find = True
                    name_mapping.update({name: bpy_name})
                    break
            assert find, 'Do not find objects in blender'
        return name_mapping 

    def move_objects(self, name_mapping:dict, pose_dict: dict):
        for key, value in pose_dict.items():
            obj = bpy.data.objects[name_mapping[key]]
            obj.rotation_mode = 'QUATERNION'
            # obj.location[0:2] = value[0:2]
            obj.location[0] = value[0]
            obj.location[1] = value[1]
            obj.location[2] = value[2]
            obj.rotation_quaternion = value[3:7]
            obj.hide_render = False

    def move_back(self, name_mapping):
        for bpy_name in name_mapping.values():
            obj = bpy.data.objects[bpy_name]
            obj.location[2] = -10
            obj.hide_render = True

    def render(self, pathdir, scene):
        for i in range(len(CAMERA_POSE)):
        # for i in range(1):
            print(i)
            pose = CAMERA_POSE[i]
            self.scanner.location[0:3] = pose[0:3]
            self.scanner.rotation_quaternion[0:4] = pose[3:7]

            save_path = get_resource_dir_path(os.path.join(pathdir, 'rendered_simple', scene))
            filename = os.path.join(save_path, "scene_{}_view_{}.pcd".format(scene, i+1))
            blensor.kinect.scan_advanced(self.scanner, evd_file=filename)
            print('Done the {} scene {} view.'.format(scene, i+1))
            # pdb.set_trace()


    def run_single_scene(self, pathdir, name_list, pose_dict, scene):

        name_mapping = self.import_objects(pathdir, name_list, scene)
        self.move_back(name_mapping)
        self.move_objects(name_mapping, pose_dict)
        self.render(pathdir, scene)

    def run(self):
        # for i, scene in enumerate(sorted(os.listdir(data_path))):
        for data in data_file_path:
            index = os.listdir(os.path.join('/data1/cxg7/models', data.split('/')[-1], 'obj'))
            pathdir = os.path.join('/data1/cxg7/models', data.split('/')[-1])
            # SAVE_DIR = os.path.join('/data/cxg13/models', data.split('/')[-1], 'rendered_simple')
            # if not os.path.exists(SAVE_DIR):
            #     os.mkdir(SAVE_DIR)
            # for scene in index:
            for scene in range(1, 151):
                scene = str(scene).zfill(5)
                print(scene)
                tic = time.time()
                name_list, _, position = read_state_from_json(os.path.join(data, scene))
                pose_list = []
                for j in range(len(position)):
                    pose_list.append(position[j][0:7])
                pose_list = np.array(pose_list)
                pose_dict = dict(zip(name_list, pose_list))
                self.run_single_scene(pathdir, name_list, pose_dict, scene)
                print("Finish {} with {}s".format(scene, time.time() - tic))
                # pdb.set_trace()




if __name__ == '__main__':
    # blensor assets/table_cycles.blend --python render/cycles_render.py
    # If you use xvfb-run, you should avoid use alias like blensor, specify the path otherwise
    # e.g.
    # xvfb-run -a -s "-screen 0 640x480x24"
    # /root/software/blensor/blender assets/table_cycles.blend --python render/cycles_render.py
    server = BlensorSceneServer()
    server.run()
import open3d
import os
from configs.torch_precomputed_single_view_point_cloud import TorchDataScenePointCloud, \
    TorchPrecomputedSingleViewPointCloud
from configs import config, path, dataset_config
from transformation_utils import local_to_global_transformation_quat
import pickle
from time import time
import numpy as np
import json
import random
import pdb

camera_pose_quat = [local_to_global_transformation_quat(pose[3:7], pose[0: 3]) for pose in config.CAMERA_POSE]

# TRAINING_DATA_MULTI_PATH = '/data/cxg13/models/eval_data'
# TRAINING_DATA_MULTI_PATH_test = '/data/cxg13/models/test'
# TRAINING_DATA_MULTI_PATH_seen_val = '/data/cxg13/models/seen_val'
# TRAINING_DATA_MULTI_PATH_seen_test = '/data/cxg13/models/seen_test'
# TRAINING_DATA_MULTI_PATH_seenval = '/data/cxg13/models/seenval_supermini'

dataset_path = '/data1/cxg7/models/file'
state_file_path = '/data1/cxg7/models/file'
data_test = os.path.join(dataset_path, 'test')
data_seen_val = os.path.join(dataset_path, 'seen_val')
data_seen_test = os.path.join(dataset_path, 'seen_test')
data_seenval = os.path.join(dataset_path, 'seenval_supermini')
data_unseen = os.path.join(dataset_path, 'unseen_val')
data_file_path = [data_test, data_seen_val ]

noise = True

def generate_single_scene(TRAINING_DATA_MULTI_PATH, scene_name, num_of_view = 9):
    current = time()

    # name_list, _, _ = dataset_config.read_state_from_json(os.path.join(data_path, scene_name))

    if type(scene_name) == int:
        scene_name = str(scene_name)
    assert type(scene_name) == str
  
    scene_path = os.path.join(TRAINING_DATA_MULTI_PATH.replace('eval_data', 'data_scene'), "{}.p".format(scene_name))
    single_view_list = [os.path.join(TRAINING_DATA_MULTI_PATH.replace('eval_data', 'rendered_simple'), scene_name, "scene_{}_view_{}.pcd".format(scene_name, view_index+1)) for view_index in range(num_of_view)]
    if noise:
        noise_single_view_list = [
            os.path.join(TRAINING_DATA_MULTI_PATH.replace('eval_data', 'rendered_simple'), scene_name, "scene_{}_view_{}_noisy.pcd".format(scene_name, view_index+1)) for view_index in range(num_of_view)]
    else:
        noise_single_view_list = None

    # scene_path, single_view_list, noise_single_view_list = path.get_data_scene_and_view_path(scene_name,
    #                                                                                          num_of_view=num_of_view,
    #                                                                                          noise=True)

    # scene_path = '/data/cxg13/models/data_scene/00001.p'
    # single_view_list = '/data/cxg13/models/rendered_simple/00001/scene_00001_view_1.pcd'
    # noise_single_view_list = '/data/cxg13/models/rendered_simple/00001/scene_00001_view_1_noisy.pcd'

    # print(scene_path, single_view_list, noise_single_view_list)
    if not (os.path.exists(scene_path) and os.path.exists(single_view_list[0])):
        print("Scene name or view name do not exist for {}".format(scene_name))
        print(scene_path)
        return

    scene_cloud = TorchDataScenePointCloud(scene_path)

    for i in range(num_of_view):
        if not os.path.exists(single_view_list[i]):
            print('NOT EXISTS THE PCD FILE!')
            continue

        output_dir = os.path.join(TRAINING_DATA_MULTI_PATH, scene_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_path = os.path.join(output_dir, "{}_view_{}.p".format(scene_name, i+1))
        if os.path.exists(output_path):
            continue
        view_cloud = TorchPrecomputedSingleViewPointCloud(open3d.io.read_point_cloud(single_view_list[i]),
                                                          open3d.io.read_point_cloud(noise_single_view_list[i]),
                                                          camera_pose=camera_pose_quat[i])

        # open3d.visualization.draw_geometries([scene_cloud.cloud, view_cloud.cloud])
        view_cloud.run_score(scene_cloud)

        # output_dir = os.path.join(TRAINING_DATA_MULTI_PATH, scene_name)
        # if not os.path.exists(output_dir):
        #     os.mkdir(output_dir)
        # output_path = os.path.join(output_dir, "{}_view_{}.p".format(scene_name, i+1))
        # pose_dir = os.path.join('/data1/cxg7/models/grasppose', scene_name)
        # if not os.path.exists(pose_dir):
        #     os.mkdir(pose_dir)
        # pose_path = os.path.join(pose_dir, '{}_view_{}.json'.format(scene_name, i+1))

        with open(output_path, 'wb') as file:
            pickle.dump(view_cloud.dump(scene_cloud), file)
            print('Save data in {}'.format(output_path)) 
        # with open(pose_path, 'w') as file:
        #     json.dump(view_cloud.save_pose(), file)
        #     print('Save data in {}'.format(pose_path))   
        # pdb.set_trace()

    print("It takes {} to finish scene {}".format(time() - current, scene_name))


def generate():
    for data in data_file_path:
        index = os.listdir(os.path.join('/data1/cxg7/models', data.split('/')[-1], 'obj'))
        TRAINING_DATA_MULTI_PATH = os.path.join('/data1/cxg7/models', data.split('/')[-1], 'eval_data')
        if not os.path.exists(TRAINING_DATA_MULTI_PATH):
            os.mkdir(TRAINING_DATA_MULTI_PATH)
        # for scene in range(210, 46514)[::-1]:
        # for scene in index:
        for scene in range(1, 151):
            scene = str(scene).zfill(5)
            print(data, scene)
            generate_single_scene(TRAINING_DATA_MULTI_PATH, scene)
            # pdb.set_trace()

if __name__ == '__main__':
    import argparse
    current_time = time()
    generate()
    print("Finished with {}s".format(-current_time + time()))

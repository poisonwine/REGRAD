import os
import socket
import pdb

hostname = socket.gethostname()


def get_resource_dir_path(resource_type: str):
    dirname = os.path.dirname(__file__)
    if hostname.startswith('grasp') or hostname.startswith('py'):
        source_dir_name = '/cephfs/dataset/ycb_data'
    else:
        source_dir_name = '/data1/cxg7/models'
    path = os.path.abspath(os.path.join(source_dir_name, resource_type))
    if not os.path.exists(path):
        # os.mkdir(path)
        os.makedirs(path)
    return path

def get_data_scene_and_view_path(scene_name: str, num_of_view, noise: bool):
    scene_path = os.path.join(DATA_SCENE_PATH, "{}.p".format(scene_name))
    view_path_list = [os.path.join(SINGLE_VIEW_PATH, scene_name, "scene_{}_view_{}.pcd".format(scene_name, view_index+1)) for
                      view_index in range(num_of_view)]
    if noise:
        noise_view_path_list = [
            os.path.join(SINGLE_VIEW_PATH, scene_name, "scene_{}_view_{}_noisy.pcd".format(scene_name, view_index+1)) for #noise
            view_index in range(num_of_view)]
    else:
        noise_view_path_list = None

    return scene_path, view_path_list, noise_view_path_list

DATA_SCENE_PATH = '/data1/cxg7/models/data_scene'
SINGLE_VIEW_PATH = '/data1/cxg7/models/rendered_simple'
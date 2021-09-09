import open3d
import numpy as np
from configs import config
from pointcloud import PointCloud
import torch
from configs.path import get_resource_dir_path
import os
import pdb

data_scene_path = get_resource_dir_path('data_scene')


class TorchDataScenePointCloud(PointCloud):
    def __init__(self, data_path, visualization=False):
        data = np.load(data_path, allow_pickle=True)
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(data['cloud'])
        PointCloud.__init__(self, cloud, visualization=visualization)

        cloud_array = data['cloud']
        normal_array = data['normal']

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        self.cloud_array_homo = torch.cat(
            [torch.tensor(cloud_array.T).float(), torch.ones(1, cloud_array.shape[0])], dim=0).float().to(device)
        self.normal_array = torch.tensor(normal_array.T).float().to(device)
        self.label_array = torch.tensor(data['label']).int().to(device)
        assert self.cloud_array_homo.shape[1] == self.normal_array.shape[1], 'shape1: {}, shape2:{}'.format(
            self.cloud_array_homo.shape, self.normal_array.shape)

        self.kd_tree = open3d.geometry.KDTreeFlann(self.cloud)
        self.normals = normal_array
        self.frame = data['frame']
        self.search_score = data['search_score']
        self.inv_search_score = data['inv_search_score']
        self.antipodal_score = data['antipodal_score']
        self.inv_antipodal_score = data['inv_antipodal_score']

        self.cloud_array_homo_table = torch.cat(
            [torch.tensor(data['cloud_table'].T).float(), torch.ones(1, data['cloud_table'].shape[0])], dim=0).float().to(device)
        
        cloud_with_table = open3d.geometry.PointCloud()
        cloud_with_table.points = open3d.utility.Vector3dVector(data['cloud_table'])
        self.kd_tree_table = open3d.geometry.KDTreeFlann(cloud_with_table)
        self.label_array_table = torch.tensor(data['label_table']).int().to(device)
        #pdb.set_trace()
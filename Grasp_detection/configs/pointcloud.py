import numpy as np
import open3d
import pdb

from configs import config
# from open3d.open3d.geometry import voxel_down_sample, estimate_normals, orient_normals_towards_camera_location

class PointCloud:
    def __init__(self, cloud: open3d.geometry.PointCloud, visualization=False):
        self.cloud = cloud
        self.visualization = visualization
        self.normals = None

    def filter_work_space(self, workspace=config.WORKSPACE):
        has_normal = self.cloud.has_normals()
        min_bound = np.array([[workspace[0]], [workspace[2]], [workspace[4]]])
        max_bound = np.array([[workspace[1]], [workspace[3]], [workspace[5]]])
        points = np.asarray(self.cloud.points)
        valid_index = np.logical_and.reduce((
            points[:, 0] > min_bound[0, 0], points[:, 1] > min_bound[1, 0], points[:, 2] > min_bound[2, 0],
            points[:, 0] < max_bound[0, 0], points[:, 1] < max_bound[1, 0], points[:, 2] < max_bound[2, 0],
        ))
        points = points[valid_index, :]
        self.cloud.points = open3d.utility.Vector3dVector(points)
        if self.cloud.has_colors():
            color = np.asarray(self.cloud.colors)[valid_index, :]
            self.cloud.colors = open3d.utility.Vector3dVector(color)
        if has_normal:
            normal = np.asarray(self.cloud.normals)[valid_index, :]
            self.cloud.normals = open3d.utility.Vector3dVector(normal)

        if self.visualization:
            open3d.visualization.draw_geometries([self.cloud])

        return valid_index

    def remove_outliers(self):
        num_points_threshold = config.NUM_POINTS_THRESHOLD
        radius_threshold = config.RADIUS_THRESHOLD
        self.cloud.remove_radius_outlier(nb_points=num_points_threshold,
                                         radius=radius_threshold)
        if self.visualization:
            open3d.visualization.draw_geometries([self.cloud])

    def voxelize(self, voxel_size=config.VOXEL_SIZE):
        #self.cloud.voxel_down_sample(voxel_size=voxel_size)
        self.cloud = open3d.geometry.PointCloud.voxel_down_sample(self.cloud, voxel_size=voxel_size)
        if self.visualization:
            open3d.visualization.draw_geometries([self.cloud])

    def estimate_normals(cloud, camera_pos=np.zeros(3), visualization = False):
        normal_radius = config.NORMAL_RADIUS
        normal_max_nn = config.NORMAL_MAX_NN
        #cloud.estimate_normals(
        #    search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
        #    fast_normal_computation=False)
        open3d.geometry.PointCloud.estimate_normals(self.cloud,
            search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
            #fast_normal_computation=False)
        self.cloud.normalize_normals()
        if True:
            open3d.geometry.PointCloud.orient_normals_towards_camera_location(self.cloud, camera_pos)
            #cloud.orient_normals_towards_camera_location(camera_pos)
        if visualization:
            open3d.visualization.draw_geometries([self.cloud])

        self.normals = np.asarray(self.cloud.normals)

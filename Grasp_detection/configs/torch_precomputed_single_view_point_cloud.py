import open3d
# from configs.dataset_config import NAME_LIST
import numpy as np
from configs import config
from pointcloud import PointCloud
from configs.torch_data_scene_point_cloud import TorchDataScenePointCloud
from tqdm import tqdm, trange
# from open3d.open3d.geometry import voxel_down_sample_and_trace, radius_outlier_removal, orient_normals_towards_camera_location
import pdb

import torch

DEBUG = False


class TorchPrecomputedSingleViewPointCloud(PointCloud):
    def __init__(self,
                 cloud: open3d.geometry.PointCloud,
                 noise_cloud: open3d.geometry.PointCloud,
                 camera_pose,
                 visualization=False):
        """
        Point Cloud Class for single view points, which is rendered before training or captured by rgbd camera
        Note that we maintain a self.frame_indices to keep a subset of all points which will calculate frame and grading
        However, all of the points should have score, and of course those do not grading will have a 0 grade
        These setting will benefit the training since we want the network to learn to distinguish table and objects

        :param cloud: Base Open3D PointCloud, no pre-process is needed ofr this input, just read from file
        :param visualization: Whether to visualize
        """

        assert np.asarray(cloud.points).shape == np.asarray(noise_cloud.points).shape
        PointCloud.__init__(self, noise_cloud, visualization)
        self.reference_cloud = np.asarray(cloud.points)
        print(self.reference_cloud.shape)

        self.index_in_ref = self.processing_and_trace()


        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        # self.name_list = name_list
        self.camera_pos = camera_pose
        self.camera_pos_torch_inv = torch.tensor(np.linalg.inv(camera_pose), device=self.device).float()

        self.cloud_array = np.asarray(self.cloud.points).astype(np.float32)
        self.cloud_color_array = np.asarray(self.cloud.colors).astype(np.float32)
        self.cloud_label = torch.zeros([len(self.cloud_array)])
        
        self.normals = np.zeros(self.cloud_array.shape, dtype=self.cloud_array.dtype)
        length_num = len(config.LENGTH_SEARCH)
        self.valid_grasp = int(0)

        self.left_normal = torch.tensor([[0, 1, 0]], device=self.device).float()
        self.right_normal = torch.tensor([[0, -1, 0]], device=self.device).float()

        self.frame_indices = None
        self.global_to_local = None
        self.valid_frame = None
        self.valid_index = None
        self.frame = None
        self.local_to_global = None
        self.frame_antipodal_score = None
        self.frame_search_score = None
        self.frame_objects_label = None
        self.pre_antipodal_score = None
        self.pre_search_score = None
        self.all_frame_bool = None

        self.antipodal_score  = None
        self.vertical_score   = None
        self.center_score_new = None
        self.score            = None
        self.best_frame       = None
        self.best_grasp_score           = None
        self.best_grasp_antipodal_score = None
        self.best_grasp_center_score    = None
        self.best_grasp_vertical_score  = None
        self.best_frame_label           = None

    @property
    def points_num(self):
        return self.cloud_array.shape[0]

    def processing_and_trace(self):
        # Filter workspace
        workspace = config.WORKSPACE
        has_normal = self.cloud.has_normals()
        has_color = self.cloud.has_colors()
        min_bound = np.array([[workspace[0]], [workspace[2]], [workspace[4]]])
        max_bound = np.array([[workspace[1]], [workspace[3]], [workspace[5]]])
        points = np.asarray(self.cloud.points)
        points_num = points.shape[0]
        valid_index = np.logical_and.reduce((
            points[:, 0] > min_bound[0, 0], points[:, 1] > min_bound[1, 0], points[:, 2] > min_bound[2, 0],
            points[:, 0] < max_bound[0, 0], points[:, 1] < max_bound[1, 0], points[:, 2] < max_bound[2, 0],
        ))
        points = points[valid_index, :]
        self.cloud.points = open3d.utility.Vector3dVector(points)
        if has_color:
            color = np.asarray(self.cloud.colors)[valid_index, :]
            self.cloud.colors = open3d.utility.Vector3dVector(color)

        if has_normal:
            normals = np.asarray(self.cloud.normals)[valid_index, :]
            self.cloud.normals = open3d.utility.Vector3dVector(normals)

        # Voxelize
        cloud_and_index = self.cloud.voxel_down_sample_and_trace(config.VOXEL_SIZE, 
                                                                    [-0.5, -0.5, 0.7], [0.5, 0.5, 1.5])
        self.cloud, trace_index_voxel = cloud_and_index[0], cloud_and_index[1]
        valid_index2 = np.max(trace_index_voxel, axis=1)

        # Remove outlier
        self.cloud, valid_index3 = self.cloud.remove_radius_outlier(config.NUM_POINTS_THRESHOLD,
        #                                                             config.RADIUS_THRESHOLD)
        # self.cloud, trace_index_voxel = voxel_down_sample_and_trace(self.cloud, config.VOXEL_SIZE, 
        #                                                             [-0.5, -0.5, 0.7], [0.5, 0.5, 1.5])
        # valid_index2 = np.max(trace_index_voxel, axis=1)

        # # Remove outlier
        # self.cloud, valid_index3 = radius_outlier_removal(self.cloud, config.NUM_POINTS_THRESHOLD,
                                                                    config.RADIUS_THRESHOLD)
                                                                    
        index_in_ref = np.arange(points_num)[valid_index][valid_index2][valid_index3]

        return index_in_ref

    def to_torch(self):
        possible_frame_num = len(self.frame_indices)
        length_num = len(config.LENGTH_SEARCH)

        self.global_to_local = torch.eye(4).unsqueeze(0).expand(possible_frame_num, 4, 4).to(self.device)
        self.valid_frame = torch.zeros(possible_frame_num, len(config.LENGTH_SEARCH), config.GRASP_PER_LENGTH,
                                       4, 4, device=self.device).float()
        self.valid_index = torch.zeros(possible_frame_num, device=self.device).int()

        self.frame_search_score = torch.zeros(possible_frame_num, length_num, config.GRASP_PER_LENGTH,
                                              dtype=torch.int,
                                              device=self.device)
        self.frame_antipodal_score = torch.zeros(possible_frame_num, length_num, config.GRASP_PER_LENGTH,
                                                 dtype=torch.float32,
                                                 device=self.device)
        self.frame_objects_label = torch.ones(possible_frame_num, length_num, config.GRASP_PER_LENGTH,
                                              dtype=torch.int16,
                                              device=self.device) * int(-1)  #len(NAME_LIST)

        self.frame = torch.tensor(self.frame).float().to(self.device).contiguous()
        self.cloud_array = torch.tensor(self.cloud_array).float().to(self.device)
        self.global_to_local = self.global_to_local.contiguous()
        self.global_to_local[:, 0:3, 0:3] = self.frame.transpose(1, 2)
        self.global_to_local[:, 0:3, 3:4] = -torch.bmm(self.frame.transpose(1, 2),
                                                       self.cloud_array[self.frame_indices, :].unsqueeze(2))
        
        self.local_to_global = torch.inverse(self.global_to_local).contiguous()
        self.frame_indices = torch.tensor(self.frame_indices, dtype=torch.int, device=self.device)

        self.pre_antipodal_score = torch.tensor(self.pre_antipodal_score, dtype=torch.float, device=self.device)
        self.pre_search_score = torch.tensor(self.pre_search_score, dtype=torch.float, device=self.device)

    def _find_match(self, scene: TorchDataScenePointCloud):
        single_frame_bool = np.zeros([self.cloud_array.shape[0]], dtype=np.bool)
        kd_tree = scene.kd_tree
        kd_tree_table = scene.kd_tree_table
        frame = np.tile(np.eye(3), [self.cloud_array.shape[0], 1, 1])
        antipodal_score = np.zeros([self.cloud_array.shape[0], len(config.LENGTH_SEARCH), config.GRASP_PER_LENGTH])
        inv_antipodal_score = np.zeros(antipodal_score.shape)
        search_score = np.zeros(antipodal_score.shape)
        inv_search_score = np.zeros(antipodal_score.shape)

        for index in range(self.cloud_array.shape[0]):
            [k, idx, _] = kd_tree_table.search_knn_vector_3d(self.reference_cloud[self.index_in_ref[index],:], 1)
            if k < 1:
                continue
            self.cloud_label[index] = scene.label_array_table[idx[0]]

        for index in range(self.cloud_array.shape[0]):
            [k, idx, _] = kd_tree.search_hybrid_vector_3d(self.reference_cloud[self.index_in_ref[index], :],
                                                          radius=config.CURVATURE_RADIUS,
                                                          max_nn=1)
            if k < 1:
                self.normals[index, :] = [0, 0, 1]
                single_frame_bool[index] = False
                continue

            i = idx[0]

            frame[index, :, :] = scene.frame[i, :, :]
            self.normals[index, :] = scene.normals[i, :]
            single_frame_bool[index] = True
            antipodal_score[index, :, :] = scene.antipodal_score[i, :, :]
            inv_antipodal_score[index, :, :] = scene.inv_antipodal_score[i, :, :]
            search_score[index, :, :] = scene.search_score[i, :, :]
            inv_search_score[index, :, :] = scene.inv_search_score[i, :, :]

        self.cloud.normals = open3d.utility.Vector3dVector(self.normals)
        self.cloud.normalize_normals()
        # self.cloud.orient_normals_towards_camera_location(camera_location=self.camera_pos[0:3, 3])
        open3d.geometry.PointCloud.orient_normals_towards_camera_location(self.cloud, camera_location=self.camera_pos[0:3, 3])
        self.normals = np.asarray(self.cloud.normals)

        self.frame = frame
        bool_frame = np.sum(self.normals * self.frame[:, :, 0], axis=1) > 0
        print('Wrong frame direction correction: {}/{}'.format(np.sum(bool_frame), bool_frame.shape[0]))

        self.frame[bool_frame, :, 0:2] = -self.frame[bool_frame, :, 0:2]
        self.pre_search_score = search_score
        self.pre_search_score[bool_frame, :, :] = inv_search_score[bool_frame, :, :]
        self.pre_antipodal_score = antipodal_score
        self.pre_antipodal_score[bool_frame, :, :] = inv_antipodal_score[bool_frame, :, :]

        possible_frame_indices, self.all_frame_bool = self.magic_formula(single_frame_bool)
        sample_frame_indices = np.arange(self.points_num)[self.cloud_array[:, 2] > config.SAMPLE_REGION]
        self.frame_indices = np.intersect1d(possible_frame_indices, sample_frame_indices)
        self.frame = frame[self.frame_indices, :, :]
        self.pre_search_score = self.pre_search_score[self.frame_indices, :, :]
        self.pre_antipodal_score = self.pre_antipodal_score[self.frame_indices, :, :]
        self.all_frame_bool = self.all_frame_bool[self.frame_indices, :, :]

    def magic_formula(self, single_frame_bool):
        search_bool = self.pre_search_score > 50
        antipodal_bool = self.pre_antipodal_score > 0.0 #0.3
        all_frame_bool = np.logical_and(search_bool, antipodal_bool)
        frame_bool = np.logical_and(all_frame_bool.max(2).max(1), single_frame_bool)
        return np.nonzero(frame_bool), all_frame_bool

    def _antipodal_score(self, close_region_cloud, close_region_cloud_normal, index: int, frame_num: int):
        """
        Estimate the antipodal score of a single grasp using scene point cloud
        Antipodal score is proportional to the reciprocal of friction angle
        Antipodal score is also divided by the square of objects in the closing region
        :param close_region_cloud: The point cloud in the gripper closing region, torch.tensor (3, n)
        :param close_region_cloud_normal: The point cloud normal in the gripper closing region, torch.tensor (3, n)
        :param index: The index of graded frame in self.cloud_array
        :param frame_num: The frame num of the point index above, to grade self.frame_antipodal_score[index, frame_num]
        :return close_region_cloud_normal for calculating projection
        """

        assert close_region_cloud.shape == close_region_cloud_normal.shape, \
            "Points and corresponding normals should have same shape"

        left_y = torch.max(close_region_cloud[1, :])
        right_y = torch.min(close_region_cloud[1, :])
        normal_search_depth = torch.min((left_y - right_y) / 3, config.NEIGHBOR_DEPTH)

        left_region_bool = close_region_cloud[1, :] > left_y - normal_search_depth
        right_region_bool = close_region_cloud[1, :] < right_y + normal_search_depth
        left_normal_theta = torch.abs(
            torch.matmul(self.left_normal, close_region_cloud_normal[:, left_region_bool]))
        right_normal_theta = torch.abs(
            torch.matmul(self.right_normal, close_region_cloud_normal[:, right_region_bool]))

        geometry_average_theta = torch.mean(left_normal_theta) * torch.mean(right_normal_theta)

        # Label based grading
        self.frame_antipodal_score[
            self.valid_grasp, frame_num // config.GRASP_PER_LENGTH, frame_num % config.GRASP_PER_LENGTH] = geometry_average_theta

    def run_score(self, scene: TorchDataScenePointCloud):
        """
        Running the main loop for all points in the class: including frame estimation and grading
        If the mode is eval, the scene point cloud is the same as itself
        :param scene: The scene point cloud used for grading
        """
        self._find_match(scene)
        # open3d.visualization.draw_geometries([self.cloud])
        self.to_torch()
        if self.frame_indices.shape[0] == 0:
            import pdb
            pdb.set_trace()

        for frame_index in trange(self.frame_indices.shape[0]):
            index = self.frame_indices[frame_index]
            if self.finger_hand(index, frame_index, scene):
                self.valid_grasp += 1

        if self.valid_grasp == 0:
            pass
        else:
            self.update_score(scene)
        
        # self.update_score(scene)

    def update_score(self, scene):
        def two_points_distance(points1, points2, points1_obj, points2_obj):
            distance = -2 * points1[:, :3].mm(points2.transpose(1,0))
            distance += torch.sum(points2.mul(points2), 1).view(1,-1).repeat(points1.size()[0],1)
            distance += torch.sum(points1[:, :3].mul(points1[:, :3]), 1).view(-1,1).repeat(1, points2.size()[0])
            
            len1 = len(points1_obj)
            len2 = len(points2_obj)
            points1_obj = points1_obj.repeat(len2,1).transpose(1,0).contiguous()
            points2_obj = points2_obj.repeat(len1,1)
            obj_mask = (points1_obj == points2_obj).double()
            return distance.double(), obj_mask

        def distance_mask(distance):
            dis_mask = (1-100*distance).mul((1-100*distance).gt(0).double())
            return dis_mask

        def compute_point_score(distance, score, obj_mask):
            dis_mask = distance_mask(distance)
            mask = dis_mask.mul(obj_mask)
            p_score = mask.mm(score.view(-1,1))
            return p_score

        scene_label = np.array(scene.label_array_table.cpu())
        scene_cloud = scene.cloud_array_homo_table[0:3, :].cpu().numpy().T
        scene_labels = np.unique(scene_label)
        object_center = {}
        object_length = {}
        for i in scene_labels:
            label_points = scene_cloud[scene_label == i]
            label_center = np.mean(label_points, axis=0)
            object_center[int(i)] = label_center
        
        for i in scene_labels:
            label_points = scene_cloud[scene_label == i]
            label_center = object_center[int(i)] 
            label_dis = np.linalg.norm((label_points[:,:2] - label_center[:2].reshape(1,-1)), axis=1)
            object_length[int(i)] = label_dis.max()
        # print(object_length)

        frame = self.valid_frame[0:self.valid_grasp, :, :, :, :].cpu().numpy()
        frame_label = self.frame_objects_label[0:self.valid_grasp, :, :].cpu().numpy().reshape(-1)
        A,B,C,_,_ = frame.shape
        frame_object_center = np.zeros((len(frame_label), 3))
        frame_object_length = np.zeros((len(frame_label)))
        for i in object_center.keys():
            frame_object_center[frame_label==i] = object_center[i]

        self.vertical_score = np.array(self.com_z_score_torch(torch.tensor(frame).view(-1,4,4)).view(A,B,C))
        frame_center = frame[...,:3,3].reshape(-1,3) + frame[...,:3,0].reshape(-1,3) * 0.06
        center_score = np.linalg.norm((frame_center[:,:2] - frame_object_center[:,:2]), axis=1).reshape(A,B,C)
        self.center_score_new = center_score.copy().reshape(-1)
        for i in object_center.keys():
            self.center_score_new[frame_label==i] /= object_length[i]
        self.center_score_new = 1-self.center_score_new.reshape(A,B,C)
        self.center_score_new = (self.center_score_new-self.center_score_new.min()) / \
                                    (self.center_score_new.max()-self.center_score_new.min())

        self.antipodal_score = self.frame_antipodal_score[0:self.valid_grasp, :, :].cpu().numpy()
        scored_grasp = (self.antipodal_score + self.center_score_new + self.vertical_score)
        
        best_frame_index = np.zeros((len(frame)))
        best_frame = np.zeros((len(frame), 4, 4))
        best_grasp_score = np.zeros((len(frame)))
        best_grasp_antipodal_score = np.zeros((len(frame)))
        best_grasp_center_score = np.zeros((len(frame)))
        best_grasp_vertical_score = np.zeros((len(frame)))
        best_frame_label = np.zeros((len(frame)))
        for i in range(len(frame)):
            f1 = frame[i].reshape(-1,4,4)
            local_search_num = np.argmax(scored_grasp[i].flatten())
            best_frame_index[i] = local_search_num
            best_frame[i] = f1[local_search_num]
            best_grasp_score[i] = scored_grasp[i].flatten()[local_search_num]
            best_grasp_antipodal_score[i] = self.antipodal_score[i].flatten()[local_search_num]
            best_grasp_center_score[i]    = self.center_score_new[i].flatten()[local_search_num]
            best_grasp_vertical_score[i]  = self.vertical_score[i].flatten()[local_search_num]
            best_frame_label[i] = frame_label.reshape(A, -1)[i,local_search_num]

        not_table_frame = (best_frame_label!=int(-1))
        best_frame_index = best_frame_index[not_table_frame]
        self.best_frame_label = best_frame_label[not_table_frame]
        self.best_frame = best_frame[not_table_frame]
        self.best_grasp_score = best_grasp_score[not_table_frame]
        self.best_grasp_antipodal_score = best_grasp_antipodal_score[not_table_frame]
        self.best_grasp_center_score    = best_grasp_center_score[not_table_frame]
        self.best_grasp_vertical_score  = best_grasp_vertical_score[not_table_frame]

        frame_label_number = {}
        for label in scene_labels:
            frame_label_number[label] = np.sum(self.best_frame_label == label)
        # print(frame_label_number)
        centers = self.best_frame[:,:3,3]
        x       = self.best_frame[:,:3,0]
        centers = centers + x * 0.06
        # view, centers = self.cloud_array.double().cuda(), torch.from_numpy(centers).double().cuda()
        # best_grasp_score = torch.from_numpy(self.best_grasp_score).double().cuda()
        # distance, obj_mask = two_points_distance(view, centers, self.cloud_label.double(), torch.from_numpy(self.best_frame_label).double())
        # score = np.array(compute_point_score(distance.cuda(), best_grasp_score, obj_mask.cuda()).view(-1).cpu())
        view, centers = self.cloud_array.double(), torch.from_numpy(centers).double()
        best_grasp_score = torch.from_numpy(self.best_grasp_score).double()
        distance, obj_mask = two_points_distance(view, centers, self.cloud_label.double(), torch.from_numpy(self.best_frame_label).double())
        score = np.array(compute_point_score(distance, best_grasp_score, obj_mask).view(-1))

        for label in scene_labels:
            if frame_label_number[label] != 0:
                score[np.array((self.cloud_label==label).cpu(), dtype=np.bool)] /= float(frame_label_number[label])
        self.score = np.tanh(score)
        print("score_max",self.score.max())
        print("score_mean",self.score.mean())


    def com_z_score_torch(self, grasp):
        grasp_z = torch.bmm(grasp[:,:3,:3].float(), torch.tensor([[1,0,0]]).float().repeat(len(grasp),1).view(-1,3,1)).transpose(2,1)
        z = torch.tensor([[0,0,-1]]).repeat(len(grasp),1).view(-1,3,1).float()
        score_angle = torch.abs(torch.bmm(grasp_z, z)).view(-1) / torch.norm(grasp_z, dim =2).view(-1)
        score_angle = 1 - torch.acos(score_angle)/np.pi # [0,1]
        return score_angle
    
    def dump(self, scene):
        """
        Damp the data inside this object to a python dictionary
        May need to store color later
        :return: Python dictionary store useful data
        """
        cloud_array = torch.cat(
            [self.cloud_array, torch.ones([self.cloud_array.shape[0], 1], dtype=torch.float, device=self.device), ],
            dim=1)
        cloud_array = torch.matmul(self.camera_pos_torch_inv, cloud_array.transpose(0, 1))
        valid_frame = self.valid_frame[0:self.valid_grasp, :, :, :, :]
        #             torch.matmul(self.camera_pos_torch_inv.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        #                           self.valid_frame[0:self.valid_grasp, :, :, :, :])
        print(valid_frame.shape)
        print(self.valid_grasp)
        result_dict = {'search_score': self.frame_search_score[0:self.valid_grasp, :, :].cpu().numpy(),
                       'antipodal_score': self.frame_antipodal_score[0:self.valid_grasp, :, :].cpu().numpy(),
                       'objects_label': self.frame_objects_label[0:self.valid_grasp, :, :].cpu().numpy(),
                       'point_cloud': cloud_array[0:3, ].cpu().numpy(),

                       'view_cloud': self.cloud_array.cpu().numpy(),
                       'view_cloud_color': self.cloud_color_array,
                       'view_cloud_label': self.cloud_label.numpy(),
                       'view_cloud_score': self.score,
                       
                       'scene_cloud': scene.cloud_array_homo[0:3, :].cpu().numpy().T,
                       'scene_cloud_table': scene.cloud_array_homo_table[0:3, :].cpu().numpy().T,

                       'valid_index': self.valid_index[0:self.valid_grasp].cpu().numpy(),
                       'valid_frame': valid_frame.cpu().numpy(),
                       
                       'vertical_score': self.vertical_score,
                       'center_score': self.center_score_new,

                       'select_frame'    : self.best_frame,
                       'select_score'    : self.best_grasp_score,
                       'select_antipodal_score'  : self.best_grasp_antipodal_score,
                       'select_center_score'     : self.best_grasp_center_score,
                       'select_vertical_score'   : self.best_grasp_vertical_score,
                       'select_frame_label'   : self.best_frame_label
                    }

        return result_dict
    
    def save_pose(self):
        print(np.unique((self.cloud_label.numpy())))
        obj_label = np.unique((self.cloud_label.numpy()))
        obj_pose = {}
        for obj in obj_label:
            if obj in set(self.best_frame_label):
                obj_frame = self.best_frame[self.best_frame_label == obj].tolist()
            else:
                obj_frame = []
            obj_pose[str(obj)] = obj_frame
        return obj_pose

    def _table_collision_check(self, point, frame):
        """
        Check whether the gripper collide with the table top with offset
        :param point: torch.tensor(3)
        :param frame: torch.tensor(3, 3)
        :return: a torch boolean tensor with shape (len(config.INDEX_TO_ARRAY))
        """

        T_local_to_global = torch.eye(4, device=self.device).float()
        T_local_to_global[0:3, 0:3] = frame
        T_local_to_global[0:3, 3] = point
        T_local_search_to_global_all = torch.bmm(
            T_local_to_global.unsqueeze(0).expand(config.LOCAL_SEARCH_TO_LOCAL.shape[0], 4, 4).contiguous(),
            config.LOCAL_SEARCH_TO_LOCAL)
        boundary_global = torch.bmm(T_local_search_to_global_all, config.TORCH_GRIPPER_BOUND.squeeze(0).expand(
            T_local_search_to_global_all.shape[0], -1, -1).contiguous())
        table_collision_bool_all = boundary_global[:, 2, :] < config.TABLE_HEIGHT - 0.005#+ config.TABLE_COLLISION_OFFSET
        return table_collision_bool_all.any(dim=1, keepdim=False)

    def finger_hand(self, index, frame_index, scene: TorchDataScenePointCloud):
        """
        Local search one point and store the closing region point num of each configurations
        Search height first, then width, finally theta
        Save the number of points in the close region if the grasp do not fail in local search
        Save the score of antipodal_grasp, note that multi-objects heuristic is also stored here
        :param index: The index of point in all single view cloud
        :param frame_index: The index of point in frame list, which is a subset of all point cloud
        :param scene: The complete scene point cloud generated to give a score
        """
        # if index == 8325:
        #     print('debug')
        #     a = 1
        frame_bool = self.all_frame_bool[frame_index, :, :]

        frame = self.frame[frame_index, :, :]
        point = self.cloud_array[index, :]

        if torch.mean(torch.abs(frame)) < 1e-6:
            return False

        table_collision_bool = self._table_collision_check(point, frame)

        T_global_to_local = self.global_to_local[frame_index, :, :]
        local_cloud = torch.matmul(T_global_to_local, scene.cloud_array_homo_table)

        i = 0
        for dl_num, dl in enumerate(config.LENGTH_SEARCH):
            if not np.any(frame_bool[dl_num, :]):
                i += config.GRASP_PER_LENGTH
                continue

            close_plane_bool = (local_cloud[0, :] < dl + config.FINGER_LENGTH) & (
                    local_cloud[0, :] > dl - config.BOTTOM_LENGTH)

            if torch.sum(close_plane_bool) < config.NUM_POINTS_THRESHOLD:
                i += config.GRASP_PER_LENGTH
                continue

            close_plane_points = local_cloud[:, close_plane_bool]  # only filter along x axis

            T_local_to_local_search_all = config.LOCAL_TO_LOCAL_SEARCH[
                                          dl_num * config.GRASP_PER_LENGTH:(dl_num + 1) * config.GRASP_PER_LENGTH,
                                          :, :]

            local_search_close_plane_points_all = torch.matmul(T_local_to_local_search_all.contiguous().view(-1, 4),
                                                               close_plane_points).contiguous().view(
                config.GRASP_PER_LENGTH, 4, -1)[:, 0:3, :]

            for _ in range(config.GRASP_PER_LENGTH):
                if table_collision_bool[i] or not frame_bool[dl_num, i % config.GRASP_PER_LENGTH]:
                    i += 1
                    continue

                local_search_close_plane_points = local_search_close_plane_points_all[i % config.GRASP_PER_LENGTH,
                                                  :, :]

                z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS) & \
                                   (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS)

                back_collision_bool = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                                      (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                                      (local_search_close_plane_points[0, :] < -config.BACK_COLLISION_MARGIN) & \
                                      z_collision_bool

                if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
                    i += 1
                    continue

                y_finger_region_bool_left = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                                            (local_search_close_plane_points[1, :] > config.HALF_BOTTOM_SPACE)
                y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                                             (local_search_close_plane_points[1, :] < -config.HALF_BOTTOM_SPACE)

                y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
                collision_region_bool = (z_collision_bool & y_finger_region_bool)
                if torch.sum(collision_region_bool) > config.FINGER_COLLISION_THRESHOLD:
                    i += 1
                    continue
                else:
                    close_region_bool = z_collision_bool & \
                                        (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_SPACE) & \
                                        (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_SPACE)

                    close_region_point_num = torch.sum(close_region_bool)

                    if close_region_point_num < config.CLOSE_REGION_MIN_POINTS:
                        # TODO: bugs in object data generation that mean is not acceptable
                        i += 1
                        continue
                    close_region_label = torch.unique(scene.label_array_table[close_plane_bool][close_region_bool],
                                                      sorted=False)
                    # print(close_region_label)
                    if int(-1) in close_region_label: # table's LABLE   len(NAME_LIST)
                        close_region_label = close_region_label[close_region_label != int(-1)]   #len(NAME_LIST)
                        # print(close_region_label)

                    if close_region_label.shape[0] > 1:
                        # print(close_region_label)
                        i += 1
                        continue

                    self.frame_search_score[
                        self.valid_grasp, i // config.GRASP_PER_LENGTH, i % config.GRASP_PER_LENGTH] = \
                        self.pre_search_score[frame_index, i // config.GRASP_PER_LENGTH, i % config.GRASP_PER_LENGTH]
                    self.frame_objects_label[
                        self.valid_grasp, i // config.GRASP_PER_LENGTH, i % config.GRASP_PER_LENGTH] = \
                        close_region_label[0]
                    self.frame_antipodal_score[
                        self.valid_grasp, i // config.GRASP_PER_LENGTH, i % config.GRASP_PER_LENGTH] = \
                        self.pre_antipodal_score[frame_index, i // config.GRASP_PER_LENGTH, i % config.GRASP_PER_LENGTH]
                    i += 1

        if torch.max(self.frame_search_score[self.valid_grasp]) < 1 or torch.max(
                self.frame_antipodal_score[self.valid_grasp]) < 0.1:
            return False

        local_to_global = self.local_to_global[frame_index, :, :]

        frame_all = torch.bmm(
            local_to_global.unsqueeze(0).expand(len(config.INDEX_TO_ARRAY), 4, 4).contiguous(),
            config.TORCH_LOCAL_SEARCH_TO_LOCAL)
        # if  self.valid_grasp == 0:
        #     print(local_to_global)
        #     print(self.global_to_local[:, 0:3, 3:4][frame_index])
        #     # print(self.cloud_array[self.frame_indices, :].unsqueeze(2)[frame_index])
        #     print(self.frame.transpose(1, 2)[frame_index])
        

        self.valid_frame[self.valid_grasp, :, :, :] = frame_all.contiguous().view(len(config.LENGTH_SEARCH),
                                                                                  config.GRASP_PER_LENGTH, 4, 4)
        self.valid_index[self.valid_grasp] = index
        return True

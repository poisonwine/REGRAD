import numpy as np
import pickle
import open3d
from time import time
import os
# from configs.dataset_config import NAME_LIST, NAME_TO_COLOR
from configs.dataset_config import read_state_from_json
from configs.path import get_resource_dir_path
from configs import config
import torch
from tqdm import trange
import pdb
from PIL import ImageColor
import transforms3d

from open3d.open3d.geometry import voxel_down_sample, sample_points_uniformly, orient_normals_towards_camera_location, estimate_normals

# path_name = "multi_width/" + str(config.WIDETH_LIST[config.number]) + "/single_object_data"
# single_object_data_path = get_resource_dir_path('single_object_data')
single_object_data_path = get_resource_dir_path('single_object_data')
#ply_dir = get_resource_dir_path('bad_ply')
ply_dir = get_resource_dir_path('ply')
# npy_dir = get_resource_dir_path('npy')
data_scene_path = get_resource_dir_path('data_scene')
table_path = '/data/cxg13/grasp_detection/table.ply'

dataset_path = '/data/cxg8/cleaned_3DVMRD/train/'
color_map = np.array([ImageColor.getrgb(color) for color in ImageColor.colormap.keys()]) / 255

class GenerateDarbouxObjectData:
    def __init__(self):
        self.ply_dir = ply_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        print(self.device)
        self.left_normal = torch.tensor([[0, 1, 0]], device=self.device).float()
        self.right_normal = torch.tensor([[0, -1, 0]], device=self.device).float()
        self.table = open3d.io.read_point_cloud(table_path)

    def dump(self, data, name, scene):
        save_path = get_resource_dir_path(os.path.join(single_object_data_path, scene))
        with open(os.path.join(save_path,'{}.p'.format(name)), 'wb') as f:
            pickle.dump(data, f)
        print('Dump to file of {} with keys: {}'.format(name, data.keys()))

    def run_loop(self):
        def global_to_local_transformation(frame, point):
            T_global_to_local = np.eye(4)
            T_global_to_local[0:3, 0:3] = frame.T
            T_global_to_local[0:3, 3] = -np.dot(frame.T, point)
            return T_global_to_local
        for j, scene in enumerate(sorted(os.listdir(dataset_path))):
            print(scene)
            file_path = os.path.join(data_scene_path, "{}.p".format(scene))
            if os.path.exists(file_path):
                continue
            name_list, _, pose = read_state_from_json(os.path.join(dataset_path, scene))
            # pose_dict = dict(zip(name_list, position))
            final_frame = []
            final_inv_frame = []

            final_cloud = []
            final_label = []
            final_color = []
            final_normal = []

            final_cloud_table = [np.asarray(self.table.points)]
            final_normal_table = [np.asarray(self.table.normals)]
            color_table = color_map[-1]
            color_array = np.tile(color_table, [len(final_cloud_table[0]), 1])
            label = int(-1)
            label_array = np.ones(len(final_cloud_table[0])) * label
            final_color_table = [color_array]
            final_label_table = [label_array]

            final_search_score = []
            final_inv_search_score = []
            final_antipodal_score = []
            final_inv_antipodal_score = []
            graspss = []

            for i, name in enumerate(name_list):
                tic = time()
                data_dict = {}
                ply_path = os.path.join(self.ply_dir, scene, "{}.ply".format(name))
                mesh = open3d.io.read_triangle_mesh(ply_path)
                mesh.compute_vertex_normals()
                pc = sample_points_uniformly(mesh, np.asarray(mesh.vertices).shape[0] * 10)
                pc = voxel_down_sample(pc, 0.0025)
                estimate_normals(pc, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                obj_id = name.split('-')[-1]
                num = int(obj_id) * 3
                color = color_map[num]
                pc.paint_uniform_color(color)
                #pc.orient_normals_towards_camera_location(camera_location=np.mean(np.asarray(pc.points), axis=0))
                # orient_normals_towards_camera_location(pc, camera_location=np.mean(np.asarray(pc.points), axis=0))
                normals = np.asarray(pc.normals)
                normals /= -np.linalg.norm(normals, axis=1, keepdims=True)
                kd_tree = open3d.geometry.KDTreeFlann(pc)

                points = np.asarray(pc.points)
                index = np.random.choice(points.shape[0], int(len(points)/10), replace = False)
                frames, inv_frames = self._estimate_frame(points[index], normals, kd_tree)
                data_dict.update({'cloud': np.asarray(pc.points)[index]})
                data_dict.update({'normal': normals[index], 'frame': frames, 'inv_frame': inv_frames})

                search_score, inv_search_score, antipodal_score, inv_antipodal_score, grasp_data = \
                            self._estimate_grasp_quality(points, points[index], frames, inv_frames, normals)
                data_dict.update(grasp_data)


                # search_point = np.sum(search_score > 0, axis=2).sum(axis=1)
                # inv_search_point = np.sum(inv_search_score > 0, axis=2).sum(axis=1)
                # better_than_inverse_bool = search_point > inv_search_point
                # score1 = inv_search_score
                # score1[better_than_inverse_bool, :, :] = search_score[better_than_inverse_bool]
                # score2 = inv_antipodal_score
                # score2[better_than_inverse_bool, :, :] = antipodal_score[better_than_inverse_bool]
                # scored_grasp = np.minimum(np.log(score1 + 1) / 4, np.ones([1, 1, 1])) * score2 \
                #                      / np.sqrt(1 - np.power(score2, 2))
                # grasps = np.full((len(frames),4,4), -1.0)
                
                # for j in range(len(frames)):
                #     T_global_to_local = global_to_local_transformation(frames[j], np.asarray(pc.points)[index][j, :])
                #     # print(T_global_to_local)
                #     local_search_num = np.argmax(scored_grasp[j].flatten())
                #     T_local_to_local_search = config.LOCAL_TO_LOCAL_SEARCH[local_search_num].cpu().numpy()
                #     T_global_to_local_search = T_local_to_local_search @ T_global_to_local
                #     # print(T_global_to_local_search)
                #     grasps[j] = T_global_to_local_search
                # print(grasps)

                print("Finish {} with time: {}s".format(name, time() - tic))
                self.dump(data_dict, name, scene)

                rotation = transforms3d.quaternions.quat2mat(pose[i][3:7])
                translation = pose[i][0:3]
                mat = np.eye(4)
                mat[0:3, 0:3] = rotation
                mat[0:3, 3] = translation

                pc_homo = np.concatenate([points[index], np.ones([points[index].shape[0], 1])], axis=1).T  # (4,n)
                # grasp_after_move = np.matmul(mat, grasps)
                # graspss.append(grasp_after_move)
                pc_after_move = np.dot(mat, pc_homo)
                final_cloud.append(pc_after_move[0:3, :].T)
                final_cloud_table.append(pc_after_move[0:3, :].T)

                normal_after_move = np.dot(rotation, normals[index].T).T
                final_normal.append(normal_after_move)
                final_normal_table.append(normal_after_move)

                label = int(obj_id)
                label_array = np.ones(pc_homo.shape[1]) * label
                final_label.append(label_array)
                final_label_table.append(label_array)

                # frame = self.frame[name]
                # inv_frame = self.inv_frame[name]
                frame_after_move = np.matmul(rotation, frames)
                final_frame.append(frame_after_move)
                inv_frame_after_move = np.matmul(rotation, inv_frames)
                final_inv_frame.append(inv_frame_after_move)

                color_array = np.tile(color, [pc_homo.shape[1], 1])
                final_color.append(color_array)
                final_color_table.append(color_array)

                final_search_score.append(search_score)
                final_inv_search_score.append(inv_search_score)
                final_antipodal_score.append(antipodal_score)
                final_inv_antipodal_score.append(inv_antipodal_score)
                # print(final_search_score)
                # print(graspss)
            clouds = np.concatenate(final_cloud, axis=0)
            # graspss = np.concatenate(graspss, axis=0)
            frames = np.concatenate(final_frame, axis=0)
            inv_frames = np.concatenate(final_inv_frame, axis=0)
            colors = np.concatenate(final_color, axis=0)
            labels = np.concatenate(final_label, axis=0)
            normals = np.concatenate(final_normal, axis=0)
            search_scores = np.concatenate(final_search_score, axis=0)
            antipodal_scores = np.concatenate(final_antipodal_score, axis=0)
            inv_search_scores = np.concatenate(final_inv_search_score, axis=0)
            inv_antipodal_scores = np.concatenate(final_inv_antipodal_score, axis=0)

            clouds_table = np.concatenate(final_cloud_table, axis=0)
            colors_table = np.concatenate(final_color_table, axis=0)
            labels_table = np.concatenate(final_label_table, axis=0)
            normals_table = np.concatenate(final_normal_table, axis=0)

            data = {'cloud': clouds, 'cloud_table': clouds_table, 'label': labels, 'label_table': labels_table,
                    'color': colors, 'color_table': colors_table, 'normal': normals, 'normal_table': normals_table,
                    'frame': frames, 'inv_frame': inv_frames, 'search_score': search_scores, 'antipodal_score': antipodal_scores,
                    'inv_search_score': inv_search_scores, 'inv_antipodal_score': inv_antipodal_scores}  #'grasps': graspss

            # file_path = os.path.join(data_scene_path, "{}.p".format(scene))
            # if os.path.exists(file_path):
            #     continue
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print("Finish data {} with time {}s".format(scene, time() - tic))
            # pdb.set_trace()

    def _estimate_frame(self, points, normals, kd_tree):
        """
        Estimate the Darboux frame of single point
        In self.frame, each column of one point frame is a vec3, with the order of x, y, z axis
        Note there is a minus sign of the whole frame, which means that x is the negative direction of normal
        """

        frames = np.tile(np.eye(3), [points.shape[0], 1, 1])
        inv_frames = np.tile(np.eye(3), [points.shape[0], 1, 1])
        for i in range(frames.shape[0]):
            [k, idx, _] = kd_tree.search_radius_vector_3d(points[i, :], config.CURVATURE_RADIUS)
            normal = np.mean(normals[idx, :], axis=0)
            normal /= np.linalg.norm(normal)
            if k < 5:
                frames[i, :, :] = np.zeros([3, 3])
                inv_frames[i, :, :] = np.zeros([3, 3])
                continue

            M = np.eye(3) - normal.T @ normal
            xyz_centroid = np.mean(M @ normals[idx, :].T, axis=1, keepdims=True)
            normal_diff = normals[idx, :].T - xyz_centroid
            cov = normal_diff @ normal_diff.T
            eig_value, eig_vec = np.linalg.eigh(cov)

            minor_curvature = eig_vec[:, 0] - eig_vec[:, 0] @ normal.T * np.squeeze(normal)
            minor_curvature /= np.linalg.norm(minor_curvature)
            principal_curvature = np.cross(minor_curvature, np.squeeze(normal))

            frames[i, :, :] = np.stack([-normal, -principal_curvature, minor_curvature], axis=1)
            inv_frames[i, :, :] = np.stack([normal, principal_curvature, minor_curvature], axis=1)
        return frames, inv_frames

    def _estimate_grasp_quality(self, points: np.ndarray, points_select: np.ndarray, frames: np.ndarray, inv_frames: np.ndarray,
                                normals: np.ndarray):
                                
        torch_points_select = torch.tensor(points_select, device=self.device).float().transpose(1, 0)
        torch_points_homo_select = torch.cat([torch_points_select,
                                       torch.ones(1, torch_points_select.shape[1], dtype=torch.float, device=self.device)],
                                      dim=0)

        torch_points = torch.tensor(points, device=self.device).float().transpose(1, 0)
        torch_points_homo = torch.cat([torch_points,
                                       torch.ones(1, torch_points.shape[1], dtype=torch.float, device=self.device)],
                                      dim=0)
        torch_normals = torch.tensor(normals, device=self.device).float().transpose(1, 0)

        torch_frames = torch.tensor(frames, device=self.device).float()
        torch_inv_frames = torch.tensor(inv_frames, device=self.device).float()
        frame_search_scores = torch.zeros([frames.shape[0], len(config.LENGTH_SEARCH), config.GRASP_PER_LENGTH],
                                          dtype=torch.int, device=self.device)
        inv_frame_search_scores = frame_search_scores.new_zeros(size=frame_search_scores.shape)
        frame_antipodal_score = torch.zeros(frame_search_scores.shape, dtype=torch.float, device=self.device)
        inv_frame_antipodal_score = frame_antipodal_score.new_zeros(size=frame_antipodal_score.shape)

        view_point_cloud = open3d.geometry.PointCloud()
        view_point_cloud.points = open3d.utility.Vector3dVector(points)
        vis_list = [view_point_cloud]

        print(torch_normals)

        for index in trange(torch_frames.shape[0]):
            search_scores, antipodal_score = self.finger_hand_view(index, torch_points_homo_select, \
                                                                    torch_frames[index, :, :],\
                                                                    torch_normals, torch_points_homo)

            inv_search_scores, inv_antipodal_score = self.finger_hand_view(index, torch_points_homo_select,
                                                                           torch_inv_frames[index, :, :],
                                                                           torch_normals, torch_points_homo)

            # grasp = np.eye(4)
            # grasp[:3,:3] = frames[index, :, :].numpy()
            # grasp[:3,3]  = 
            # hand = get_hand_geometry(, color=[0.5, 0, 0])
            # open3d.visualization.draw_geometries(vis_list)


            frame_search_scores[index, :, :] = search_scores
            inv_frame_search_scores[index, :, :] = inv_search_scores
            frame_antipodal_score[index, :, :] = antipodal_score
            inv_frame_antipodal_score[index, :, :] = inv_antipodal_score

        # All return variable should be numpy on the end of this function
        grasp_data = {}
        grasp_data.update({"search_score": frame_search_scores.cpu().numpy()})
        grasp_data.update({"inv_search_score": inv_frame_search_scores.cpu().numpy()})
        grasp_data.update({"antipodal_score": frame_antipodal_score.cpu().numpy()})
        grasp_data.update({"inv_antipodal_score": inv_frame_antipodal_score.cpu().numpy()})
        return frame_search_scores.cpu().numpy(), inv_frame_search_scores.cpu().numpy(),\
            frame_antipodal_score.cpu().numpy(), inv_frame_antipodal_score.cpu().numpy(), grasp_data       #grasp_data

    def finger_hand_view(self, index, point_homo_select: torch.Tensor, single_frame: torch.Tensor, normal: torch.Tensor, point_homo: torch.Tensor):
        all_frame_search_score = torch.zeros([len(config.LENGTH_SEARCH), config.GRASP_PER_LENGTH], dtype=torch.float,
                                             device=self.device)
        all_frame_antipodal = torch.zeros(all_frame_search_score.shape, dtype=torch.float, device=self.device)

        if torch.mean(torch.abs(single_frame)) < 1e-6:
            return all_frame_search_score, all_frame_antipodal

        point = point_homo_select[0:3, index:index + 1]  # (3,1)
        T_global_to_local = torch.eye(4, device=self.device).float()
        T_global_to_local[0:3, 0:3] = single_frame
        T_global_to_local[0:3, 3:4] = point
        T_global_to_local = torch.inverse(T_global_to_local)

        local_cloud = torch.matmul(T_global_to_local, point_homo)
        local_cloud_normal = torch.matmul(T_global_to_local[0:3, 0:3], normal)

        i = 0
        for dl_num, dl in enumerate(config.LENGTH_SEARCH):
            close_plane_bool = (local_cloud[0, :] < dl + config.FINGER_LENGTH) & (
                    local_cloud[0, :] > dl - config.BOTTOM_LENGTH)

            if torch.sum(close_plane_bool) < config.NUM_POINTS_THRESHOLD:
                i += config.GRASP_PER_LENGTH
                continue

            close_plane_points = local_cloud[:, close_plane_bool]  # only filter along x axis

            T_local_to_local_search_all = config.LOCAL_TO_LOCAL_SEARCH[
                                          dl_num * config.GRASP_PER_LENGTH:(dl_num + 1) * config.GRASP_PER_LENGTH, :, :]

            local_search_close_plane_points_all = torch.matmul(T_local_to_local_search_all.contiguous().view(-1, 4),
                                                               close_plane_points).contiguous().view(
                config.GRASP_PER_LENGTH, 4, -1)[:, 0:3, :]

            for _ in range(config.GRASP_PER_LENGTH):
                local_search_close_plane_points = local_search_close_plane_points_all[i % config.GRASP_PER_LENGTH, :, :]

                back_collision_bool_xy = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                                         (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                                         (local_search_close_plane_points[0, :] < -config.BACK_COLLISION_MARGIN)

                y_finger_region_bool_left = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                                            (local_search_close_plane_points[1, :] > config.HALF_BOTTOM_SPACE)
                y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                                             (local_search_close_plane_points[1, :] < -config.HALF_BOTTOM_SPACE)

                # Here we use the average of dz local search to compensate for the error in end-effector
                temp_search, temp_antipodal = torch.zeros(1, dtype=torch.float, device=self.device), torch.zeros(1, dtype=torch.float, device=self.device)
                close_region_point_num, single_antipodal = torch.zeros(1, dtype=torch.float, device=self.device), torch.zeros(1, dtype=torch.float, device=self.device)
                for dz_num, dz in enumerate([-0.02, 0.02, 0]):

                    z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS + dz) & \
                                       (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS + dz)

                    back_collision_bool = back_collision_bool_xy & z_collision_bool

                    if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
                        continue

                    y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
                    collision_region_bool = (z_collision_bool & y_finger_region_bool)
                    if torch.sum(collision_region_bool) > config.FINGER_COLLISION_THRESHOLD:
                        continue

                    close_region_bool = z_collision_bool & \
                                        (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_SPACE) & \
                                        (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_SPACE)

                    close_region_point_num = torch.sum(close_region_bool, dtype=torch.float)
                    if close_region_point_num < config.CLOSE_REGION_MIN_POINTS:
                        continue

                    xyz_plane_normals = local_cloud_normal[:, close_plane_bool][:, close_region_bool]  # (3,n)
                    close_region_cloud_normal = torch.matmul(
                        T_local_to_local_search_all[i % config.GRASP_PER_LENGTH, 0:3, 0:3],
                        xyz_plane_normals)

                    close_region_cloud = local_search_close_plane_points[:, close_region_bool]

                    single_antipodal = self._antipodal_score(close_region_cloud, close_region_cloud_normal)
                    temp_antipodal += single_antipodal / 3
                    temp_search += close_region_point_num / 3

                all_frame_search_score[dl_num, i % config.GRASP_PER_LENGTH] = torch.min(temp_search,
                                                                                        close_region_point_num)

                all_frame_antipodal[dl_num, i % config.GRASP_PER_LENGTH] = torch.min(temp_antipodal, single_antipodal)
                i += 1

        return all_frame_search_score, all_frame_antipodal

    def _antipodal_score(self, close_region_cloud, close_region_cloud_normal):
        """
        Estimate the antipodal score of a single grasp using scene point cloud
        Antipodal score is proportional to the reciprocal of friction angle
        Antipodal score is also divided by the square of objects in the closing region
        :param close_region_cloud: The point cloud in the gripper closing region, torch.tensor (3, n)
        :param close_region_cloud_normal: The point cloud normal in the gripper closing region, torch.tensor (3, n)
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
        return geometry_average_theta

if __name__ == '__main__':
    a = GenerateDarbouxObjectData()
    a.run_loop()
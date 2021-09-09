import numpy as np
import open3d
import os
from mathutils import Matrix, Vector, Euler, Quaternion
from transforms3d.euler import euler2mat
from configs.visualization_utils import get_hand_geometry, get_hand_rectangle
from configs.path import get_resource_dir_path
import pdb
import cv2
import json
import math

data_dir = dataset_path = '/data/cxg8/cleaned_3DVMRD'
output_dir = get_resource_dir_path('grasp_image')
output_file = get_resource_dir_path('grasp_file')

cam_intrisnic = np.array([1791.3843994140625, 0.0, 640.0,\
                         0.0,1791.3843994140625, 480.0, \
                         0.0, 0.0, 1.0]).reshape((3,3))


locations=[[0.8, 0, 2.1],
               [0.8, -0.8, 2.1],
               [0.8, 0.8, 2.1],
               [-0.8, 0, 2.1],
               [-0.8, 0.8, 2.1],
               [-0.8, -0.8, 2.1],
               [0, -0.8, 2.1],
               [0, 0, 2.1],
               [0, 0.8, 2.1],
               ]
points = [[-0.15, 0, 0],
          [-0.15, 0.15, 0],
          [-0.15, -0.15, 0],
          [0.15, 0, 0],
          [0.15, -0.15,0],
          [0.15, 0.15, 0],
          [0, 0.15, 0],
          [0, 0, 0],
          [0, -0.15, 0]
          ]

def angle(v1, v2):
    innerdot = abs(np.dot(v2,v1.T))
    v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
    cosalpha = innerdot/(v1_norm*v2_norm)
    return cosalpha


if __name__ == '__main__':
    # data = np.load(os.path.join(get_resource_dir_path('eval_data'), '32165.p'), allow_pickle=True)

    # for i, scene_id in enumerate(sorted(os.listdir(os.path.join(data_dir, 'train')))):
    for scene_id in range(150, 45809):
        scene_id = str(scene_id)
        if len(scene_id) != 5:
            scene_id = scene_id.zfill(5)
        for j in range(len(locations)):
            print(scene_id)
            if not os.path.exists(os.path.join(get_resource_dir_path('eval_data'), scene_id, '{}_view_{}.p'.format(scene_id, j+1))):
                import pdb
                pdb.set_trace()
            data = np.load(os.path.join(get_resource_dir_path('eval_data'), scene_id, '{}_view_{}.p'.format(scene_id, j+1)), allow_pickle=True)
            output_path = get_resource_dir_path(os.path.join(output_dir, scene_id))
            output_file_path = get_resource_dir_path(os.path.join(output_file, scene_id))
            bbox_file = os.path.join(output_file_path, '{}.json'.format(j+1))
            # if os.path.exists(bbox_file):
            #     print('HAVE DONE!')
            #     continue
            point = Vector(points[j])
            location = Vector(locations[j])
            direction = point - location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            rotation_euler = rot_quat.to_euler()
            mat = euler2mat(rotation_euler[0],rotation_euler[1], rotation_euler[2])
            camera_extrinisic = np.concatenate((mat, np.expand_dims(location,axis=0).T), axis = 1)
            camera_extrinisic = np.concatenate((camera_extrinisic, np.array([[0,0,0,1]])), axis = 0)

            image_path = os.path.join(data_dir, 'train', scene_id, str(j+1), 'rgb.jpg')
            # if not os.path.exists(image_path):
            #     continue
            image = cv2.imread(image_path)

            frame = data['select_frame']
            label = data['select_frame_label']
            view = data['view_cloud']
            view_cloud_label = data['view_cloud_label']
            vertical_score = data['select_vertical_score']
            center_score = data['select_center_score']
            antipodal_score = data['select_antipodal_score']
            score = data['select_score']
            view_point_cloud = open3d.geometry.PointCloud()
            view_point_cloud.points = open3d.Vector3dVector(view)
            center = np.array([frame[i][0:3,3] for i in range(frame.shape[0])])
            index = (vertical_score > 0.0)
            # frame = frame[index]
            # label = label[index]
            # vertical_score = vertical_score[index]
            # center_score = center_score[index]
            # antipodal_score = antipodal_score[index]
            # score = score[index]


            num = frame.shape[0]
            print(num)

            # index_2 = (label == 2.0)
            # frame = frame[index_2]

            hand_list = []
            label_select = []
            result = []
            box_list = []
            select_vertical_score, select_center_score, select_antipodal_score, select_score = [],[],[],[]


            for k in range(num):
                normal  = frame[k,0:3,0]
                center = frame[k,0:3,3]
                center2cam = center - locations[j]
                # if angle(normal, direction) < math.cos(math.pi/6):
                #     continue
                global_to_local = np.linalg.inv(frame[k, :, :])
                hand = get_hand_geometry(global_to_local, color=[0.5, 0, 0])
                hand_list.append(hand)
                label_select.append(label[k])
                select_vertical_score.append(vertical_score[k])
                select_center_score.append(center_score[k])
                select_antipodal_score.append(antipodal_score[k])
                select_score.append(score[k])

            for v in range(len(hand_list)):
                point2d = get_hand_rectangle(hand_list[v],camera_extrinisic, cam_intrisnic)
                result.append(point2d.T[:, 0:2])
            result = np.array(result)
            if result.shape[0] == 0:
                continue
            print('result.shape:', result.shape)
            print(len(label_select))
            x = result[:, :, 0]
            y = result[:, :, 1]
            assert x.shape[0] == y.shape[0]
            
            for q in range(x.shape[0]):
                cnt = np.float32(((x[q,3], y[q,3]),(x[q,1],y[q,1]), (x[q,0], y[q,0]), (x[q,2], y[q,2])))
                rect = cv2.minAreaRect(cnt)   #center of the box + width + height + angle
                box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
                box = np.int0(box)  #the vertice of the box, the type is int.
                box_list.append([label_select[q], list(rect), [select_vertical_score[q],select_center_score[q], select_antipodal_score[q], select_score[q]]])
                # box_list.append(rect)
                # box_list.append([select_vertical_score[q],select_center_score[q], select_antipodal_score[q], select_score[q]])

                # cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
                # cv2.putText(image, str('id'+str(label_select[q]).split('.')[0]), tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1) #label_select.append(label[k])

                #cv2.imshow('fff', image)
            # cv2.imwrite(os.path.join(output_path, '{}.jpg'.format(j+1)), image)
            
            with open(bbox_file, 'w') as f:
                json.dump(box_list, f)
            # print(box_list)
            pdb.set_trace()




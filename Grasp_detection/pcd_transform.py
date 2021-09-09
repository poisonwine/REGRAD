import open3d,os
import numpy as np
import pdb


# camera_to_world = [np.array([[ 0.79885968,  0.        ,  0.60151742,  0.8       ],
#                              [ 0.        ,  1.        ,  0.        ,  0.        ],
#                              [-0.60151742,  0.        ,  0.79885968,  1.7       ],
#                              [ 0.        ,  0.        ,  0.        ,  1.        ]]), 
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
                # ]

camera_to_world = [np.array([[ 7.54978995e-08, -9.11107897e-01,  4.12167926e-01, 0.8],
                            [ 1.00000000e+00,  8.46019382e-08,  3.84210991e-09, 0],
                            [-3.83707821e-08,  4.12167926e-01,  9.11107897e-01, 2.1],
                            [0,0,0,1]]), 
                   np.array([[ 7.07106724e-01, -5.95639885e-01,  3.81068509e-01, 0.8],
                            [ 7.07106839e-01,  5.95639709e-01, -3.81068571e-01, -0.8],
                            [ 1.03974664e-07,  5.38912297e-01,  8.42361879e-01, 2.1],
                            [0,0,0,1]]), 
                   np.array([[-7.07106785e-01, -5.95639764e-01,  3.81068583e-01, 0.8],
                            [ 7.07106777e-01, -5.95639784e-01,  3.81068568e-01, 0.8],
                            [ 1.67853056e-08,  5.38912348e-01,  8.42361847e-01, 2.1],
                            [0,0,0,1]]), 
                   np.array([[ 7.54978995e-08,  9.11107897e-01, -4.12167926e-01, -0.8],
                            [-1.00000000e+00,  8.46019382e-08,  3.84210991e-09, 0],
                            [ 3.83707821e-08,  4.12167926e-01,  9.11107897e-01, 2.1],
                            [0,0,0,1]]),
                    np.array([[-7.07106785e-01,  5.95639764e-01, -3.81068583e-01, -0.8],
                            [-7.07106777e-01, -5.95639784e-01,  3.81068568e-01, 0.8],
                            [-1.67853056e-08,  5.38912348e-01,  8.42361847e-01, 2.1],
                            [0,0,0,1]]),
                    np.array([[ 7.07106724e-01,  5.95639885e-01, -3.81068509e-01, -0.8],
                            [-7.07106839e-01,  5.95639709e-01, -3.81068571e-01, -0.8],
                            [-1.03974664e-07,  5.38912297e-01,  8.42361879e-01, 2.1],
                            [0,0,0,1]]),
                    np.array([[ 1.         ,-0.          ,0.        , 0],
                            [ 0.          ,0.9111079  ,-0.41216793, -0.8],
                            [ 0.          ,0.41216793  ,0.9111079, 2.1 ],
                            [0,0,0,1]]),
                    np.array([[ 1. ,-0.  ,0., 0],
                            [ 0.  ,1. ,-0., 0],
                            [ 0.  ,0.  ,1., 2.1],
                            [0,0,0,1]]),
                    np.array([[-1.00000000e+00, -1.52425057e-07,  2.94057925e-08, 0],
                            [ 1.50995799e-07, -9.11107910e-01,  4.12167899e-01, 0.8],
                            [-3.60328656e-08,  4.12167899e-01,  9.11107910e-01, 2.1],
                            [0,0,0,1]])
]


# SAVE_DIR = "/data/cxg13/models/rendered_simple/00001"
dataset_path = '/data1/cxg7/models/file'
state_file_path = '/data1/cxg7/models/file'
data_test = os.path.join(dataset_path, 'test')
data_seen_val = os.path.join(dataset_path, 'seen_val')
data_seen_test = os.path.join(dataset_path, 'seen_test')
data_seenval = os.path.join(dataset_path, 'seenval_supermini')
data_unseen = os.path.join(dataset_path, 'unseen_val')
data_file_path = [data_test, data_seen_val]

for data in data_file_path:
#     index = os.listdir(os.path.join('/data1/cxg7/models', data.split('/')[-1], 'obj'))
    for scene_name in range(1, 151):
        scene_name = str(scene_name).zfill(5)
        # for j, scene_name in enumerate(sorted(os.listdir(data_path))[::-1]):
        # SAVE_DIR = os.path.join('/data1/cxg7/models/rendered_simple', scene_name)
        SAVE_DIR = os.path.join('/data1/cxg7/models', data.split('/')[-1], 'rendered_simple', scene_name)
        for i in range(9):
            filename = os.path.join(SAVE_DIR, "scene_{}_view_{}00000.pcd".format(scene_name,  i+1))
            filename_noise = os.path.join(SAVE_DIR, "scene_{}_view_{}_noisy00000.pcd".format(scene_name, i+1))
            print(filename_noise)
            if not os.path.exists(filename):
                print('NOT EXISTS THE PCD FILE!')
                continue
            if not os.path.exists(filename_noise):
                print('NOT EXISTS THE NOISE PCD FILE!')
                continue

            formal_cloud = open3d.io.read_point_cloud(filename)
            print(len(formal_cloud.points))
            formal_noise_cloud = open3d.io.read_point_cloud(filename_noise)

            #cloud = open3d.PointCloud()
            #cloud.points = open3d.Vector3dVector(formal_cloud.points)
            cloud = formal_cloud
            cloud.transform(camera_to_world[i])

            #cloud_noise = open3d.PointCloud()
            #cloud_noise.points = open3d.Vector3dVector(formal_noise_cloud.points)
            cloud_noise = formal_noise_cloud
            cloud_noise.transform(camera_to_world[i])

            open3d.io.write_point_cloud(filename.replace("00000.pcd", ".pcd"), cloud)
            open3d.io.write_point_cloud(filename_noise.replace("00000.pcd", ".pcd"), cloud_noise)
            os.remove(filename)
            os.remove(filename_noise)
            print("delete: ", filename)
            print("delete: ", filename_noise)
            # pdb.set_trace()

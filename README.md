# REGRAD Dataset

![REGRAD](REGRAD.png)  
This is a detailed decription of REGRAD Dataset.  
The Dataset can be divided into two parts, *Relation Part*  and  *Grasp Part*. Each part contains five folder, 
*train/test/seen_valid/unseen_valid/seen_test*. For every folder, the data format is the same.  
![dataset_split](dataset_split.png) 

## Relation Part
The directory structure is as follows. Take *train* folder for example.  
```
train  
|_____ scene id  
|__________ id of different camera angles(from 1 to 9)
|________________ camera_info.json
|________________ info.json
|________________ rgb.jpg
|________________ segment.jpg
|________________ depth.png
|________________ labeled.jpg
|________________ minRect_labeled.jpg
|__________ final_state.json
|__________ mrt.json
```
### Some detailed description

- [x] *final_state.json*  
    contains the  generated scene information including *model name*、*obj_id*、*path*、*position*. This file is necessary to
    reload the scene in relation detection.
    ```angular2
    { 'name': model_name-obj_id (e.g. tower-1),
      'path': model category/model path (e.g. 04460130/de08da18d316f927a72fcffccc240663),
      'pos': xyz postion + quaternion,
  }
    ```
- [x] *mrt.json*  
    Model relation tree. Contains every model's parent list.  
    e.g.  
    ```angular2
    {"table-1": ['birdhouse-2','mug-4'], "birdhouse-2": [], "vessel-3": ['mug-4'], "mug-4": [] }
    ```
- [x] *camera_info.json*  
    contains camera extrisinc and intrisinc parameters.

- [x] *info.json*  
    contains all models information,it is a list of dictionaries. Every
    dictionary contains all information of a model.Dict keys are as follows.  
    ```angular2
    { 'model_name':(str) ShapeNet model name(e.g. tower),
  
      'category':(str) ShapeNet category(e.g. 04460130),
  
      'model_id':(str) ShapeNet model id(e.g. de08da18d316f927a72fcffccc240663),
  
      'obj_id': (int) id in the scene(e.g. 1),
  
      '6D_pose':(7-d list of float) 6D_pose of models, format [x,y,z, quaternion],
  
      'parent_list':(list) parent obj_id list, format ['model_name-obj_id1','model_name-obj_id2'],
  
      'scale':(3-d list) model scale factor in the scene, format [x_scale, y_scale, z_scale],
  
      'bbox':(4-d list) bounding box of the model, format [x1, y1, x2, y2],
  
      'MinAreaRect':(4-d list) The smallest bounding rectangle of the model, format [x1, y1, x2, y2],
  
      'segmentation':(2-d list) segmentation region in the segment.jpg picture, format [x_position_list, y_position_list],
  
      'Source': (str) ShapeNetCore.v2,
  }
  ```
- [x] *segment.jpg*  
    For better visualize, the segment id of a model can be  computed  with *10 obj_id +1*.  
   
- [x] *labeled.jpg*  
    visualized bounding box
- [x] *minRect_labeled.jpg*  
    visualized minimum bounding box
- [x] *depth.png*  


## Grasp_Part  
As for grasping, the dataset provides both 2D and 3D grasping data. For each scene, there are 9 data files
corresponding to 9 camera angles.  
1. data loading
```angular2
    import numpy as np
    data = np.load(filepath, allow_pickle=True)
```
2. data format
```angular2
{
    'antipodal_score'([selected_point_num, 4,9]): The force closure property of a grasp,
 
    'vertical_score'([selected_point_num, 4,9]): The verticality between the grasping direction of the gripper and the table (It is more stable to perform grasping perpendicular to the desktop)

    'center_score'([selected_point_num, 4,9]): The distance between the grasp point and the center point of the object (It is more stable to perform grasping near its center)
    
    'objects_label'([selected_point_num, 4,9]): The object label corresponding to the point of the effective grasping part
    
    'view_cloud'([view_point_num, 3]): Point cloud from the camera's perspective
    
    'view_cloud_color'([view_point_num, 3]): Point cloud' color from the camera's perspective
    
    'view_cloud_label'([view_point_num, ]): The object label corresponding to each point in the point cloud from the camera's perspective
    
    'view_cloud_score'([view_point_num, 4, 9]): The ratio of successful grasp in a region around a given point P, which can help learn which position in P is suitable for grasping
    
    'scene_cloud'([scene_point_num, 3]): Complete scene point cloud (excluding table)
    
    'scene_cloud_table'([scene_point_num+table_point_num, 3]): Complete scene point cloud (including table)
    
    'valid_index'([valid_grasp_num, ]): The index of the point with the effective grasping
    
    'valid_frame'([valid_grasp_num, 4, 9, 4, 4]): Effective grasping
    
    'select_frame'([valid_grasp_num, 4, 4]): Grasp with the highest score (score = antipodal_score + vertical_score + center_score)
    
    'select_score'([valid_grasp_num, ]): Max of antipodal_score + vertical_score + center_score
    
    'select_antipodal_score'([valid_grasp_num, ]): Max of antipodal_score
    
    'select_center_score'([valid_grasp_num, ]): Max of center_score
    
    'select_vertical_score'([valid_grasp_num, ]): Max of vertical_score
    
    'select_frame_label'([valid_grasp_num, ]): The label of the object corresponding to each frame in select_frame
}
```

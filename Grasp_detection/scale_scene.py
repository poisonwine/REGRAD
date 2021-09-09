import copy
import os
import time
import pdb
import json
import cv2
import bpy
import random
import sys
sys.path.append(os.getcwd())

import numpy as np
#import open3d
import transforms3d
from transforms3d.euler import quat2euler
from mathutils import Matrix, Vector, Euler, Quaternion
import bpycv
from bpycv.object_utils import activate_obj,load_obj
# from open3d.open3d.geometry import voxel_down_sample, estimate_normals

from configs.dataset_config import read_state_from_json
from configs.path import get_resource_dir_path
import pdb

dirname = os.path.dirname(__file__)

dataset_path = '/data/cxg8/cleaned_3DVMRD'
shapenet_path = '/data/zhb/ShapeNetCore.v2'
plane_back_path = '/data/cxg8/3dvmrd_v2/table_background'
table_path_dir = '/data/cxg13/models'

state_path = os.path.join(dataset_path, 'train')
data_test = os.path.join(dataset_path, 'test')
data_seen_val = os.path.join(dataset_path, 'seen_val')
data_seen_test = os.path.join(dataset_path, 'seen_test')
data_seenval = os.path.join(dataset_path, 'seenval_supermini')
data_file_path = [data_test, data_seen_val, data_seen_test, data_seenval]


# def set_plane():
#     bpy.ops.mesh.primitive_plane_add(size=1, calc_uvs=True, enter_editmode=False, align='WORLD', location=(0, 0, 0.49), rotation=(0, 0, 0))  
#     if "Cube" in bpy.data.objects.keys():
#                 bpy.data.objects.remove(bpy.data.objects["Cube"])         
#     obj = bpy.data.objects['Plane']
#     obj['inst_id'] = 255
#     bpy.data.materials.new('plane_tex')
#     mat = bpy.data.materials['plane_tex']
#     obj.data.materials.append(mat)
#     mat.use_nodes =True
#     tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
#     pic = random.choice(os.listdir(plane_back_path))
#     pic_path = os.path.join(plane_back_path, pic)
#     tex.image = bpy.data.images.load(filepath = pic_path)
#     disp = mat.node_tree.nodes['Principled BSDF'].inputs['Base Color']
#     mat.node_tree.links.new(disp,tex.outputs[0])
#     table_path_ply = os.path.join(table_path_dir, "table.ply")
#     table_path_obj = os.path.join(table_path_dir, "table.obj")
#     bpy.ops.export_mesh.ply(filepath=table_path_ply, axis_forward='X', axis_up='Z', use_colors=True)
#     bpy.ops.export_scene.obj(filepath=table_path_obj, use_selection=True, axis_forward='X', axis_up='Z')
#     #pdb.set_trace()
#     #bpy.ops.import_mesh.ply(filepath=table_path_ply)
#     #bpy.ops.mesh.select_all(action='SELECT')
#     bpy.ops.object.mode_set(mode='EDIT')
#     bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY')
#     table_path_ply_2 = os.path.join(table_path_dir, "table_2.ply")
#     bpy.ops.export_mesh.ply(filepath=table_path_ply_2, axis_forward='X', axis_up='Z', use_colors=True)
#     # bpy.ops.export_scene.obj(filepath=table_path_obj, use_selection=True, axis_forward='X', axis_up='Z')
#     bpy.data.objects.remove(bpy.data.objects['Plane'])


class ScenePrepare:
    def __init__(self, data_path, scene_id, obj_output_dir, ply_output_dir):
        self.data_path = data_path
        self.scene_id = scene_id
        self.obj_output_dir = obj_output_dir
        self.ply_output_dir = ply_output_dir
        #self.scale_objects()
        #self.set_plane()
    
    def scale_objects(self):
        name, path, position = read_state_from_json(os.path.join(self.data_path, self.scene_id))
        assert len(name) == len(path) == len(position)
        for i in range(len(name)):
            obj_path = os.path.join(shapenet_path, path[i],'models','model_normalized.obj')
            bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='X', axis_up='Z')
            if "Cube" in bpy.data.objects.keys():
                bpy.data.objects.remove(bpy.data.objects["Cube"])
            for obj in bpy.data.objects:
                if obj.name.split('.')[0] == 'model_normalized':
                    obj.name = name[i].split('-')[-1]
            object_name = [obj for obj in bpy.data.objects.keys()][0]
            print(object_name)
            bpy.data.objects[object_name].select_set(True) #2.8x
            bpy.context.view_layer.objects.active = bpy.data.objects[object_name]  # 2.8

            obj_name = name[i].split('-')[0]
            obj_id = name[i].split('-')[1]
            obj_scale = position[i][7:]
            #obj_output_dir = os.path.join(get_resource_dir_path('obj'), '{}'.format(self.scene_id)))
            #ply_output_dir = os.path.join(get_resource_dir_path('ply'), '{}'.format(self.scene_id)))
            output_path_obj = os.path.join(self.obj_output_dir, "{}.obj".format(name[i]))
            output_path_ply = os.path.join(self.ply_output_dir, "{}.ply".format(name[i]))

            bpy.data.objects[object_name].scale = (obj_scale[0], obj_scale[1], obj_scale[2])
            bpy.ops.export_scene.obj(filepath=output_path_obj, use_selection=True, axis_forward='X', axis_up='Z')
            
            bpy.ops.export_mesh.ply(filepath=output_path_ply, axis_forward='X', axis_up='Z', use_colors=True)
            bpy.data.objects.remove(bpy.data.objects[object_name])


if __name__ == '__main__':

    # Get tabel's mesh model and Just run only once
    # table = set_plane()
    
    # Get mesh model from origin model
    for data in data_file_path:
        TEST_PATH = os.path.join('/data/cxg13/models', 'test')
        if not os.path.exists(TEST_PATH):
            os.mkdir(TEST_PATH)

        for scene in os.listdir(data):
            scene = str(scene).zfill(5)
            print(scene)
            obj_output = get_resource_dir_path(os.path.join(TEST_PATH, data.split('/')[-1], 'obj', '{}'.format(scene)))
            ply_output = get_resource_dir_path(os.path.join(TEST_PATH, data.split('/')[-1], 'ply', '{}'.format(scene)))
            # if len(os.listdir(obj_output)) != 0:
            #     print('finished scene: ', scene)
            #     continue
            objects = ScenePrepare(data, scene, obj_output, ply_output)
            objects.scale_objects()
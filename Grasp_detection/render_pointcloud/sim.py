import bpy#, phobos
from bpy import data as D
from bpy import context as C
from mathutils import *
from math import *
import numpy as np
import os
import blensor

CAMERA_POSE = [
    [0.8, 0, 1.7, 0.948, 0, 0.317, 0],
    [-0.8, 0, 1.6, -0.94, 0, 0.342, 0],
    [0.0, 0.75, 1.7, 0.671, -0.224, 0.224, 0.671],
    [0.0, -0.75, 1.6, -0.658, -0.259, -0.259, 0.658]
]

def load_obj(file_number):

    bpy.ops.wm.open_mainfile(filepath="base1.blend")
    bpy.ops.import_scene.obj(filepath="table_blender.obj")
    """If the scanner is the default camera it can be accessed 
        for example by bpy.data.objects["Camera"]"""
    scanner = bpy.data.objects["Camera"]

    """Move it to a different location"""
    scanner.location = (0.844,0,1.634)
    #scanner.location = (0.8,0,1.0)
    scanner.rotation_euler = (50/180*np.pi, 0, 90/180*np.pi)
    scanner.add_scan_mesh = False
    scanner.add_noise_scan_mesh = False

    obj_object = bpy.context.scene.objects
    print(obj_object)
    pos_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), \
                              "bullet3/my_pybullet_code/pos/", str(file_number)+".txt")
    f = open(pos_path, "r")
    txt = f.readlines()
    obj = []
    pos = np.zeros((int(len(txt)/2), 3))
    ori = np.zeros((int(len(txt)/2), 4))
    for i in range(len(txt)):
        if i%2 == 0:
            obj.append(txt[i].split("\n")[0])
        else:
            number = txt[i].split("\n")[0].split(" ")
            for j in range(3):
                pos[int(i/2), j] = float(number[j])
            for j in range(4):
                ori[int(i/2), j] = float(number[j+3])
    #print(obj)
    #print(pos)
    #print(ori)

    obj_names = []
    obj_name_indexes = []
    for i in range(len(obj)):
        obj_name = obj[i]
        obj_path = os.path.join("/home/lesley/code/gym/model/ycb_meshes_google/", obj_name, "google_512k/textured_small.obj")                  
        bpy.ops.import_scene.obj(filepath=obj_path)
        cur_obj_object = bpy.context.scene.objects[0]
        #print(cur_obj_object.name)
        obj_names.append(cur_obj_object.name)
        obj_name_indexes.append(int(obj_name.split("_")[0]))

        cur_obj = bpy.data.objects[cur_obj_object.name]
        cur_obj.location = (pos[i,0], pos[i,1], pos[i,2])
        cur_obj.rotation_mode = 'QUATERNION' # (w x y z)
        cur_obj.rotation_quaternion = (ori[i,3], ori[i,0], ori[i,1], ori[i,2])
        cur_index = int(obj_name.split("_")[0])
        #'''
        for num, m in list(enumerate(cur_obj.material_slots)): #rename materials with . numbers
            print(cur_obj.name, m.name, int(obj_name.split("_")[0]))
            new_name = m.name+"_0"
            cur_obj.data.materials.clear()#.material_slot_remove()
            m_new = bpy.data.materials.new(new_name) #set new material to variable
            cur_obj.data.materials.append(m_new) #add the material to the object
            bpy.data.materials[new_name].diffuse_color = [0.01*cur_index, 1, 1] #m.material.name
        #'''

        
    obj_object = bpy.context.scene.objects
    print(obj_object)    
    point_folder = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), "bullet3/my_pybullet_code/point_color")
    if not os.path.isdir(point_folder):
        os.makedirs(point_folder)
    point_path = os.path.join(point_folder, str(file_number) + ".pcd")
    print(point_path)
    blensor.kinect.scan_advanced(scanner, evd_file=point_path)
    #blensor.blendodyne.scan_advanced(scanner, rotation_speed = 10.0, 
    #                                simulation_fps=24, angle_resolution = 0.1728, 
    #                                max_distance = 120, evd_file= "/tmp/scan1.pcd",
    #                                noise_mu=0.0, noise_sigma=0.03, start_angle = 0.0,  
    #                                end_angle = 360.0, evd_last_scan=True, 
    #                                add_blender_mesh = False, 
    #                                add_noisy_blender_mesh = False)

    '''
    for obj_name in obj_names:
        print("delete:\t",obj_name)
        obj = bpy.data.objects[obj_name]
        obj.user_clear # clear this object of users
        bpy.context.scene.objects.unlink(obj) # unlink the object from the scene
        bpy.data.objects.remove(obj)

    obj_objects = bpy.context.scene.objects
    print("saved obj :", obj_objects)
    '''
    bpy.ops.wm.read_factory_settings()#use_empty=True)


for file_number in range(207,208):
    print("----------------------The ", file_number+1, "th file------------------------------")
    load_obj(file_number)
#load_obj(0)

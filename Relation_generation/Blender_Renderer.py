import bpy
import os
import time
import random
import numpy as np
import argparse
import sys
import json
from transforms3d.euler import quat2euler
from mathutils import Matrix, Vector, Euler, Quaternion
import cv2
import bpycv
from bpycv.object_utils import activate_obj,load_obj
from scipy.stats import entropy

dataset_path = '/home/ydy/data2/3DVMRD'
shapenet_path = '/home/ydy/data/ShapeNetCore.v2'
collision_path = '/home/ydy/data/collision_obj'
table_back_path = './table_background'
world_back_path = './world_background'

class BlenderRender():
    def __init__(self, 
                data_path, 
                scene_id, 
                camera_location,
                camera_point, 
                render_save_path,
                background_random=False,
                texture_random=True,
                use_gpu = True):       
         self.data_path = data_path
         self.scene_id = scene_id
         self.location = camera_location
         self.point = camera_point
         self.save_path = render_save_path
         self.background_random = background_random
         self.texture_random = texture_random
         self.use_gpu = use_gpu
         self.set_scene()
         start = time.time()
         self.save_render_image()
         print('save  image time:', time.time()-start)

    def set_scene(self):
        bpycv.clear_all()
        bpy.context.scene.frame_set(1)
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 32
        bpy.context.view_layer.cycles.use_denoising = True
        bpy.context.view_layer.cycles.denoising_store_passes = True
        if self.use_gpu:
            bpy.context.scene.cycles.device = "GPU"
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            print("compute_device_type =",  bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
            for d in  bpy.context.preferences.addons["cycles"].preferences.devices:
                d["use"] = True
                print(d["name"], d["use"])
            print('use gpu to render')
        start1 = time.time()
        if self.background_random:
            if random.random() < 0.7:
                hdri_manager = bpycv.HdriManager(hdri_dir=world_back_path, download=False) # if download is True, will auto download .hdr file from HDRI Haven
                hdri_path = hdri_manager.sample()
                bpycv.load_hdri_world(hdri_path, random_rotate_z=True)


        self.set_plane()
        start3 = time.time()
        self.set_camera(self.location,self.point)
        self.set_light()
        self.import_objs_to_scene()
        start4 = time.time()
        print('import_obj time:',start4-start3)
        # for i in range(5):
        #     bpy.context.scene.frame_set(bpy.context.scene.frame_current+1)

    def enable_gpus(self, device_type,  device_list):
 
        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        cuda_devices, opencl_devices = cycles_preferences.get_devices()
        if device_type == "CUDA":
            devices = cuda_devices
        elif device_type == "OPENCL":
            devices = opencl_devices
        else:
            raise RuntimeError("Unsupported device type")
        activate_gpus = []
        for i, device in enumerate(devices):
            if i in device_list:
                device.use =True
                activate_gpus.append(device.name)
            else:
                device.use = False
        cycles_preferences.compute_device_type = device_type
        for scene in bpy.data.scenes:
            scene.cycles.device = "GPU" 


    def set_plane(self):
        bpy.ops.mesh.primitive_plane_add(size=1, calc_uvs=True, enter_editmode=False, align='WORLD', location=(0, 0, 0.49), rotation=(0, 0, 0))           
        obj = bpy.data.objects['Plane']
        obj['inst_id'] = 255
        bpy.data.materials.new('plane_tex')
        mat = bpy.data.materials['plane_tex']
        obj.data.materials.append(mat)
        mat.use_nodes =True
        tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
        pic = random.choice(os.listdir(table_back_path))
        pic_path = os.path.join(table_back_path, pic)
        tex.image = bpy.data.images.load(filepath = pic_path)
        disp = mat.node_tree.nodes['Principled BSDF'].inputs['Base Color']
        mat.node_tree.links.new(disp,tex.outputs[0])
    
    def set_camera(self,location, point):
        cam_intrisnic = [1791.3843994140625, 0.0, 640.0,\
                         0.0,1791.3843994140625, 480.0, \
                         0.0, 0.0, 1.0]

        cam_obj = bpy.data.objects['Camera']
        ### set cam from extrinsic
        cam_obj.rotation_mode = 'XYZ'
        cam_obj.location = Vector(location)
        point = Vector(point)
        direction = point - cam_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()

        ###set camera from intrisnic
        cam_data = cam_obj.data
        width, height = 1280, 960
        cam_data['loaded_resolution'] = [width,height]
        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height
        cam_data['loaded_intrinisics'] = cam_intrisnic
        #cam_data.lens_unit='FOV'
        cam_K = np.array(np.array(cam_intrisnic).reshape(3, 3).astype(np.float32))
        cam_data.angle = 2*np.arctan(width/(2*cam_K[0, 0]))
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy= cam_K[0, 2], cam_K[1, 2]
        if fx > fy:
            bpy.context.scene.render.pixel_aspect_y = fx/fy
        elif fx < fy:
            bpy.context.scene.render.pixel_aspect_x = fy/fx
        max_resolution = max(width, height)
        cam_data.shift_x = -(cx-width/2.0) / max_resolution
        cam_data.shift_y = (cy-height/2.0) / max_resolution
        cam_data.clip_start = 0.1
        cam_data.clip_end = 100
        cam_extrinsic = cam_obj.matrix_world
        intrinsic_matrix = np.array(cam_intrisnic).tolist()
        extrinsic_matrix = np.array(cam_extrinsic).tolist()
        camera_info = {}
        camera_info.update({'intrinisic': intrinsic_matrix})
        camera_info.update({'extrinsic': extrinsic_matrix})
        str = json.dumps(camera_info)
        with open(os.path.join(self.save_path, 'camera_info.json'), 'w') as f:
            f.write(str)


    def set_light(self):
        ###set the first light,white light
        light_data1 = bpy.data.lights.new(name='light1',type='POINT')
        light_data1.energy = np.random.uniform(200,300)
        x = np.random.uniform(0.5, 1)
        light_data1.color = [x, x, x]
        size1 = np.random.uniform(1, 2)
        light_data1.shadow_soft_size = size1
        light1 = bpy.data.objects.new(name='light1',object_data = light_data1)
        bpy.context.collection.objects.link(light1)
        loc_x , loc_y = np.random.uniform(-1.5, 1.5),np.random.uniform(-1.5, 1.5)
        loc_z = np.random.uniform(3.5, 3.5)
        light1_location= [loc_x, loc_y, loc_z]
        light1.location = light1_location
        
        #####set the second light
        light_data2 =bpy.data.lights.new(name = 'light2', type='POINT')
        light_data2.energy = np.random.uniform(150, 250)
        random_color = np.random.uniform(low=0, high=1, size=(3,))
        light_data2.color = random_color
        size2 = np.random.uniform(1, 1.5)
        light_data2.shadow_soft_size = size2
        light2 = bpy.data.objects.new(name = 'light2', object_data = light_data2)
        bpy.context.collection.objects.link(light2)
        loc_x , loc_y = np.random.uniform(0.5, 1.5),np.random.uniform(0.5, 1.5)
        loc_z = np.random.uniform(2.5, 3.5)
        light2_location= [loc_x, loc_y, loc_z]
        light2.location = light2_location

    def read_state_from_json(self):
        state_path = os.path.join(self.data_path, self.scene_id, 'final_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                result = json.load(f)
            name = []
            path = []
            position = []
            for dict in result:
                name.append(dict['name'])
                path.append(dict['path'])
                pos_str = dict['pos']
                pos = [float(s) for s in pos_str]
                position.append(pos)
        else:
            name, path, position = [], [], []
            print('final_state.json file in not exist')
        return name, path, position
        
    def import_objs_to_scene(self):
        name, path, position = self.read_state_from_json()
        assert len(name) == len(path) == len(position)
        for i in range(len(name)):
            obj_path = os.path.join(shapenet_path, path[i],'models','model_normalized.obj')
            obj = load_obj(obj_path)
            obj.location = Vector(position[i][:3])
            euler = quat2euler(position[i][3:7])
            obj.rotation_euler = euler
            obj.scale = Vector(position[i][7:])
            id = int(name[i].split('-')[-1])
            obj['inst_id'] = 10*id+1
            if self.texture_random:
                texture_randomization(obj)
        else:
            print('final_state.json file in not exist')
            
    def save_render_image(self):
        result = bpycv.render_data()
        rgb_path = os.path.join(self.save_path,'rgb.jpg')
        cv2.imwrite(rgb_path, result["image"][..., ::-1])
        instant_path = os.path.join(self.save_path,'segment.jpg')
        cv2.imwrite(instant_path, np.uint16(result["inst"]))
        depth_path = os.path.join(self.save_path, 'depth.png')
        depth_in_mm = result["depth"] * 1000
        cv2.imwrite(depth_path, np.uint16(depth_in_mm))
        vis_path = os.path.join(self.save_path, 'vis_inst_rgb_depth.jpg')
        cv2.imwrite(vis_path, result.vis()[..., ::-1])

        names, paths, pos = self.read_state_from_json()
        with open(os.path.join(self.data_path, self.scene_id, 'mrt.json'), 'r', encoding='UTF-8') as f:
            mrt = json.load(f)
        keys = ['model_name', 'category', 'model_id', 'obj_id', '6D_pose', 'parent_list', 'scale', 'bbox', 'segmentation', 'source']
        model_list = []
        obj_ids = []
        for i in range(len(names)):
            model_dict = dict.fromkeys(keys)
            model_dict['model_name'] = names[i].split('-')[0]
            model_dict['category'] = paths[i].split('/')[0]
            model_dict['model_id'] = paths[i].split('/')[1]
            model_dict['obj_id'] = int(names[i].split('-')[1]) * 10+1
            model_dict['parent_list'] = mrt[names[i]]
            model_dict['6D_pose'] = pos[i][:7]
            model_dict['scale'] = pos[i][7:]
            model_dict['source'] = 'ShapeNetCore.v2'
            model_list.append(model_dict)
            obj_ids.append(model_dict['obj_id'])

        seg = np.uint16(result['inst'])
        bounding_boxes = {}
        minAreaRect ={}
        for i in range(len(obj_ids)):
            seg_area = np.where(seg == obj_ids[i])
            seg_area = [np.array(seg_area[0]).tolist(), np.array(seg_area[1]).tolist()]
            model_list[i]['segmentation'] = seg_area
            if len(seg_area[0]) != 0:
                y_min, y_max = np.min(seg_area[0]), np.max(seg_area[0])
                x_min, x_max = np.min(seg_area[1]), np.max(seg_area[1])
                bounding_box = [x_min, y_min, x_max, y_max]
                bounding_boxes.update({names[i]: bounding_box})
                model_list[i]['bbox'] = np.array(bounding_box).tolist()
                cnt = [[seg_area[1][i], seg_area[0][i]] for i in range(len(seg_area[0]))]
                cnt = np.array(cnt)
                rect =cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                minAreaRect.update({names[i]: box})
                model_list[i]['minAreaRect'] = np.array(box).tolist()
            else:
                model_list[i]['bbox'] = None
                model_list[i]['minAreaRect'] =None
        label_rgb = cv2.imread(rgb_path)
        minRect_label_rgb = cv2.imread(rgb_path)
        for key, bbox in bounding_boxes.items():
            cv2.rectangle(label_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(label_rgb, key, (bbox[0],bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(self.save_path, 'labeled.jpg'),label_rgb)
        for key, box in minAreaRect.items():
            cv2.line(minRect_label_rgb, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0, 0, 255), 2)
            cv2.line(minRect_label_rgb, (box[1][0], box[1][1]), (box[2][0], box[2][1]), (0, 0, 255), 2)
            cv2.line(minRect_label_rgb, (box[2][0], box[2][1]), (box[3][0], box[3][1]), (0, 0, 255), 2)
            cv2.line(minRect_label_rgb, (box[3][0], box[3][1]), (box[0][0], box[0][1]), (0, 0, 255), 2)
            cv2.putText(minRect_label_rgb, key, (box[1][0],box[1][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(self.save_path, 'minRect_labeled.jpg'), minRect_label_rgb)
        str = json.dumps(model_list)
        with open(os.path.join(self.save_path, 'info.json'), 'w') as f:
            f.write(str)
        print('succeful save render image')    
                
     
def texture_randomization(obj, smooth_rate=0.8):
    if random.random() < smooth_rate:
        with activate_obj(obj):
            bpy.ops.object.shade_smooth()
        if obj.material_slots.keys() is not None:
            material = obj.material_slots[0].material
            bsdf = material.node_tree.nodes['Principled BSDF']
            bsdf.inputs["Metallic"].default_value = random.random()
            bsdf.inputs["Specular"].default_value = random.random()
            bsdf.inputs["Specular Tint"].default_value = random.random()
            bsdf.inputs["Roughness"].default_value = random.uniform(0.3, 0.6)
            bsdf.inputs["Anisotropic"].default_value = random.random()
            bsdf.inputs["Anisotropic Rotation"].default_value = random.random()
            bsdf.inputs["Sheen"].default_value = random.random()
            bsdf.inputs["Sheen Tint"].default_value = random.random()
            bsdf.inputs["Clearcoat"].default_value = random.random()
            bsdf.inputs["Clearcoat Roughness"].default_value = random.random()
            bsdf.inputs["Alpha"].default_value = random.uniform(0.9, 1)
        else:
            bpy.ops.material.new('random-tex')
            mat = bpy.data.material['random-tex']
            obj.data.materials.append(mat)
            mat.use_nodes =True
            tex = mat.node_tree.nodes.new('ShaderNodeRGB')
            rgb = np.random.uniform(0, 1, size=(3,))
            tex.outputs[0].default_value = rgb
            disp = mat.node_tree.nodes['Principled BSDF'].inputs['Base Color']
            mat.node_tree.links.new(disp,tex.outputs[0])
            

if __name__ =='__main__':
    start =time.time()
    types = 'seen_test'
    data_path = os.path.join(dataset_path, types)
    save_path = os.path.join(dataset_path, types)
    #scene_ids = os.listdir(data_path)
    scene_ids =["%05d"%int(i) for i in range(11750,15000)]
    #scene_ids = ['04377']
    #####
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

    for scene_id in scene_ids:
        print('#####################################')
        print('processing {:s}: {:d}/{:d}'.format(scene_id, scene_ids.index(scene_id), len(scene_ids)))
        print('####################################')
        path_exist = False
        if os.path.exists(os.path.join(save_path, scene_id)):
            path_exist = True
        p = os.path.join(save_path, scene_id, '9', 'info.json')
        mrt_path = os.path.join(save_path, scene_id, 'mrt.json')
        with open(mrt_path, 'r') as  f:
            mrt = json.load(f)
        state_path = os.path.join(save_path, scene_id, 'final_state.json')
        with open(state_path, 'r') as  f:
            states = json.load(f)
        flag = True
        for dic in states:
            if dic['name'] not in mrt.keys():
                flag = False
        if (not os.path.exists(p)) and flag and path_exist:
            for i in range(len(locations)):
                cam_path0 = os.path.join(save_path, scene_id, str(i+1))
                if not os.path.exists(cam_path0):
                    os.makedirs(cam_path0)
                location = locations[i]
                point = points[i]
                render = BlenderRender(data_path=data_path,
                                       scene_id=scene_id, 
                                       camera_location=location,
                                       camera_point=point,
                                       background_random=True,
                                       render_save_path=cam_path0,
                                       use_gpu=False)
                render = None
    print(time.time()-start)


import os
from PIL import Image, ImageDraw, ImageColor
# import open3d
import argparse
import random
import time
from transforms3d.euler import euler2quat
import sapien.core as sc
from sapien.core.pysapien import Pose, PxrMaterial, OptifuserRenderer, OptifuserController, SceneConfig, VulkanRenderer, VulkanController
import numpy as np
import json
import sample

# visual_model_path = '/data/zhb/ShapeNetCore.v2'
# collison_model_path = '/data/cxg8/collision_obj'
# dataset_save_path = '/data/cxg8/3DVMRD_3'

visual_model_path = '/home/ydy/data/ShapeNetCore.v2'
collison_model_path = '/home/ydy/data/collision_obj'
dataset_save_path = '/home/ydy/data2/3DVMRD'

collsion = 'collision.obj'
visual_path = 'models/model_normalized.obj'
import sys

sys.path.append(visual_model_path)
sys.path.append(collison_model_path)


def parsers():
    parser = argparse.ArgumentParser(description='3Dvmrd parameters')
    parser.add_argument('--type', type=str, default='test',
                        help='which dataset type: train, seen_val,seen_test, unseen_val, test')
    parser.add_argument('--min_size', type=float, default=0.08, help='minimum object size(m) ')
    parser.add_argument('--max_size', type=float, default=0.20, help='max size (m)')
    parser.add_argument('--min_obj', type=int, default=5, help='min object number')
    parser.add_argument('--max_obj', type=int, default=20, help='max object number')
    parser.add_argument('--gui', type=bool, default=False, help='whether to use gui')
    parser.add_argument('--start_id', type=int, default=5, help='scene start id')
    parser.add_argument('--end_id', type=int, default=10, help='scene end id')
    args = parser.parse_args()
    return args


def setup_table(scene: sc.Scene, height, table_physical_material):
    table_size = np.array([1, 1, 0.01]) / 2
    table_pose = np.array([0, 0, height - 0.01])
    table_vis_material = sc.PxrMaterial()
    table_vis_material.roughness = 0.025
    table_vis_material.specular = 0.95
    table_vis_material.metallic = 0.6
    rgbd = np.array([171, 171, 171, 255])
    table_vis_material.set_base_color(rgbd / 255)
    builder = scene.create_actor_builder()
    builder.add_box_visual_complex(sc.Pose(table_pose), table_size, table_vis_material)
    builder.add_box_shape(sc.Pose(table_pose), table_size, table_physical_material)
    table = builder.build_static("table")
    table.set_pose(sc.Pose([0, 0, 0], [-0.7071, 0, 0, 0.7071]))

    table_leg_position1 = [0.45, 0.35, height / 2]
    table_leg_position2 = [-0.45, -0.35, height / 2]
    table_leg_position3 = [-0.45, 0.35, height / 2]
    table_leg_position4 = [0.45, -0.35, height / 2]
    table_leg_size = np.array([0.025, 0.025, height / 2 - 0.01])
    builder = scene.create_actor_builder()
    builder.add_box_visual_complex(sc.Pose(table_leg_position1), table_leg_size)
    builder.add_box_visual_complex(sc.Pose(table_leg_position2), table_leg_size)
    builder.add_box_visual_complex(sc.Pose(table_leg_position3), table_leg_size)
    builder.add_box_visual_complex(sc.Pose(table_leg_position4), table_leg_size)
    legs = builder.build_static("table_leg")
    legs.set_pose(table.get_pose())

    return [table, legs]


def set_camera(scene, camera_name, position):
    near, far = 0.1, 100
    camera_mount_actor = scene.create_actor_builder().build(is_kinematic=True, name=camera_name)
    camera = scene.add_mounted_camera(camera_name, camera_mount_actor, Pose(),
                                      1280, 960, np.deg2rad(30), np.deg2rad(30), near, far)

    pos = np.array(position)
    forward = -pos / np.linalg.norm(pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.linalg.inv(np.array([forward, left, up]))
    mat44[:3, 3] = pos
    camera_mount_actor.set_pose(Pose.from_transformation_matrix(mat44))

    return camera, mat44


class object_sampler(object):
    def __init__(self,
                 args,
                 ):
        self.args = args

    def sample_objects(self):
        obj_dic = sample.get_sampled_dict(self.args.type)
        num_obj = random.randrange(self.args.min_obj, self.args.max_obj)
        with open('id2name.json', 'r') as f:
            id2name = json.load(f)
        sample_list = []
        ids = list(obj_dic.keys())
        prob_list = sample.get_sample_prob(ids)
        for i in range(num_obj):
            id_choice = sample.random_pick(ids, prob_list)
            name = id2name[id_choice]
            name_ids = obj_dic[id_choice]
            if name_ids is not None:
                name_id_choice = random.sample(name_ids, 1)[0]
                c_path = os.path.join(collison_model_path, id_choice, name_id_choice, collsion)
                if os.path.exists(c_path):
                    sample_list.append([name, id_choice, name_id_choice])

        return sample_list

    def get_scale_values(self, obj_path, min_length, max_length):
        scale_value = np.zeros((3,), dtype=float)
        # open the obj file and get the points
        with open(obj_path, 'r') as file:
            points = []
            while True:
                line = file.readline()
                if not line:
                    break
                str = line.split(" ")
                if str[0] == 'v':
                    points.append((float(str[1]), float(str[2]), float(str[3])))
        points = np.array(points)
        xyz_min, xyz_max = np.min(points, axis=0), np.max(points, axis=0)
        bounding_box = xyz_max - xyz_min
        # max_box_length = np.max(bounding_box)
        size = np.random.uniform(low=min_length, high=max_length)
        scale_value[:] = size / np.max(bounding_box)
        return list(scale_value)

    def sample_size(self, id, nameid):

        obj_path = os.path.join(visual_model_path, id, nameid, visual_path)
        min_size = self.args.min_size
        max_size = self.args.max_size
        scale_value = self.get_scale_values(obj_path, min_size, max_size)
        return scale_value

    def sample_position(self):
        x = np.random.uniform(low=-0.15, high=0.15)
        y = np.random.uniform(low=-0.15, high=0.15)
        z = np.random.uniform(low=0.8, high=0.9)
        pos = [x, y, z]
        return pos

    def sample_pose(self):
        eulerAngle = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
        quat = euler2quat(eulerAngle[0], eulerAngle[1], eulerAngle[2])
        return list(quat)


class grasp_scene(object):
    def __init__(self,
                 args,
                 ):
        # initialize the scene settings
        self.object_sampler = object_sampler(args)
        self.args = args

    def _load_one_model(self, scene, name, path, state, type):
        # load a model into the scene
        builder = scene.create_actor_builder()
        pos = state[:3]
        quat = state[3:7]
        scale = state[7:]
        builder.add_multiple_convex_shapes_from_file(
            os.path.join(collison_model_path, path, collsion),
            scale=scale)
        builder.add_visual_from_file(
            os.path.join(visual_model_path, path, visual_path),
            scale=scale)
        if type == 'dynamic':
            model = builder.build(name=name)
            model.set_pose(sc.Pose(pos, quat))
            model.set_velocity([0, 0, 0])
            model.set_damping(1, 1)

        if type == 'static':
            model = builder.build_static(name=name)
            model.set_pose(sc.Pose(pos, quat))

        return model

    def _load_scene_state_static(self, scene, name_list, path_list, state_list):
        actors = []
        assert len(name_list) == len(path_list) == len(state_list)
        for i in range(len(name_list)):
            builder = scene.create_actor_builder()
            pos = state_list[i][:3]
            quat = state_list[i][3:7]
            scale = state_list[i][7:]
            try:
                builder.add_multiple_convex_shapes_from_file(
                    os.path.join(collison_model_path, path_list[i], collsion),
                    scale=scale)
                builder.add_visual_from_file(os.path.join(visual_model_path, path_list[i], visual_path),
                                             scale=scale)
                model = builder.build_static(name=name_list[i])
                model.set_pose(sc.Pose(pos, quat))
                actors.append(model)
            except:
                pass
        scene.step()

        return actors

    def save_state(self, save_path, objs_list, final_pos_and_quat, scale):

        assert len(objs_list) == len(final_pos_and_quat) == len(scale)
        save_list = []

        for i in range(len(objs_list)):
            pos_qua = np.array(final_pos_and_quat[i]).tolist()
            scale_value = np.array(scale[i]).tolist()
            state = {}
            state.update({'name': objs_list[i][0]})
            state.update({'path': objs_list[i][1] + '/' + objs_list[i][2]})
            state.update({'pos': pos_qua + scale_value})
            save_list.append(state)
        final_state = json.dumps(save_list, indent=6)
        with open(os.path.join(save_path, 'final_state.json'), 'w') as f:
            f.write(final_state)
            f.close()

    def get_scene_save_path(self):
        scene_path = os.path.join(dataset_save_path, self.args.type)
        return scene_path

    def step_second(self, scene):
        if args.gui:
            for p in range(360):
                scene.step()
                scene.update_render()

        else:
            for p in range(360):
                scene.step()

    def step_ms(self, scene):
        if args.gui:
            for p in range(360):
                scene.step()
                scene.update_render()
        else:
            for p in range(360):
                scene.step()

    def generete_scene(self, engine, scene_id):
        table_height = 0.5
        objs_list = self.object_sampler.sample_objects()
        # create scene save dirs
        save_path = self.get_scene_save_path()
        scene_path = os.path.join(save_path, scene_id)
        if not os.path.exists(scene_path):
            os.makedirs(scene_path)
        scales = []
        names = []
        final_pos_quat = []  # a tuple which the form is (pos,quat)
        #renderer = OptifuserRenderer()
        #engine.set_renderer(renderer)
        config = SceneConfig()
        config.gravity = [0, 0, -1]
        scene = engine.create_scene(config=config)
        set_up_world(engine, scene, table_height)

        for i in range(len(objs_list)):
            obj = objs_list[i]
            builder = scene.create_actor_builder()
            # sample every object's scale position and quaternion
            scale = self.object_sampler.sample_size(obj[1], obj[2])
            pos = self.object_sampler.sample_position()
            quat = self.object_sampler.sample_pose()
            try:
                builder.add_multiple_convex_shapes_from_file(
                    os.path.join(collison_model_path, obj[1], obj[2], collsion),
                    scale=scale)
                builder.add_visual_from_file(os.path.join(visual_model_path, obj[1], obj[2], visual_path),
                                             scale=scale)

                model = builder.build(name=obj[0] + '-' + str(i))
                model.set_pose(sc.Pose(pos, quat))
                model.set_velocity([0, 0, 0])
                model.set_damping(random.uniform(0.8, 1.5), random.uniform(0.8, 1.5))
                names.append(model.get_name())
                scales.append(scale)
                for t in range(3):
                    self.step_second(scene)
            except:
                pass
        for t in range(3):
            self.step_second(scene)
        # check the model that on the desk
        actors = scene.get_all_actors()[3:]
        obj_on_desk = []
        scale_on_desk = []
        flag = 0
        for i in range(len(actors)):
            actor = actors[i]
            pos = list(actor.get_pose().p)
            # check the object whether on the table
            if pos[2] > table_height:
                new_name = objs_list[i][0] + '-' + str(flag)
                obj_on_desk.append([new_name, objs_list[i][1], objs_list[i][2]])
                scale_on_desk.append(scales[i])
                quat = list(actor.get_pose().q)
                final_pos_quat.append(list(pos) + list(quat))
                flag +=1
        # save the state
        self.save_state(scene_path, obj_on_desk, final_pos_quat, scale_on_desk)
        print('generate scene successful')
        scene = None
        # then you should generate the whole scene according to the above informations

    def gen_mrt(self, engine, scene_id):

        # generate the manipulation relationship tree for a scene
        # scene state should be a set of model states'
        print('begin to check relationship')
        table_height = 0.5
        names, paths, scene_state = self.read_final_state_from_json(scene_id)
        mrtree = {}
        #renderer = OptifuserRenderer()
        #engine.set_renderer(renderer)
        scene = engine.create_scene()
        set_up_world(engine, scene, table_height)
        for i in range(len(scene_state)):
            # print('chech the {}th object'.format(i))
            name_without_obj, path_withou_obj, state_without_obj = self.get_static_state_with_removedmodel(names, paths,
                                                                                                           scene_state,
                                                                                                           i)
            static_actor = self._load_scene_state_static(scene, name_without_obj, path_withou_obj, state_without_obj)
            parent_node = []
            for j in range(len(state_without_obj)):
                dyobj_actor = self._load_one_model(scene, names[i], paths[i], scene_state[i], 'dynamic')
                self.step_second(scene)
                pose1 = dyobj_actor.get_pose().p
                scene.remove_actor(static_actor[j])
                static_actor.pop(j)
                self.step_second(scene)
                pose2 = dyobj_actor.get_pose().p
                if self._check_pose_dif(pose1, pose2):
                    parent_node.append(name_without_obj[j])
                scene.remove_actor(dyobj_actor)
                scene.step()
                #scene.update_render()
                static_model = self._load_one_model(scene, name_without_obj[j], path_withou_obj[j],
                                                    state_without_obj[j], 'static')
                scene.step()
                #scene.update_render()
                static_actor.insert(j, static_model)

            mrtree.update({names[i]: list(parent_node)})
            print('Finish {}th object check'.format(i))
            actors = scene.get_all_actors()[3:]
            for ac in actors:
                scene.remove_actor(ac)
            scene.step()
            #scene.update_render()
        # save mrt
        mrt = json.dumps(mrtree)
        path = self.get_scene_save_path()
        scene_path = os.path.join(path, scene_id)
        with open(os.path.join(scene_path, 'mrt.json'), 'w') as json_file:
            json_file.write(mrt)
        scene = None
        print('finish checking relationship')

    def read_final_state_from_json(self, scene_id):

        scene_path = os.path.join(dataset_save_path, self.args.type, scene_id, 'final_state.json')
        with open(scene_path, 'r') as f:
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

        return name, path, position

    def get_static_state_with_removedmodel(self, name_list, path_list, pos_list, remove_id):

        assert len(name_list) == len(path_list) == len(pos_list)
        new_name_list = name_list[:]
        new_path_list = path_list[:]
        new_state_list = pos_list[:]
        del new_name_list[remove_id]
        del new_path_list[remove_id]
        del new_state_list[remove_id]
        return new_name_list, new_path_list, new_state_list

    def _check_pose_dif(self, pose1, pose2):
        pose_diff = [pose1[i] - pose2[i] for i in range(len(pose1))]
        n = 0
        for num in pose_diff:
            n += num ** 2
        thresh = 5e-6
        return True if n > thresh else False

def set_up_world(engine, scene, table_height):
    table_material = engine.create_physical_material(1.2, 0.8, 0.01)
    setup_table(scene, table_height, table_material)
    scene.set_timestep(1 / 120)
    ground_material = PxrMaterial()
    ground_color = np.array([202, 164, 114, 256]) / 256
    ground_material.set_base_color(ground_color)
    ground_material.specular = 0.5
    scene.add_ground(0.0, render_material=ground_material)



if __name__ == '__main__':
    table_height = 0.5
    args = parsers()
    engine = sc.Engine()
    scene_ids = [str(i) for i in range(args.start_id, args.end_id)]
    # config = sc.SceneConfig()
    # scene_ids = ['27','28','29']
    for scene_id in scene_ids:
        scene_path = os.path.join(dataset_save_path, args.type, scene_id, 'mrt.json')
        print('processing {:d}/{:d}'.format(scene_ids.index(scene_id), len(scene_ids)))
        if not os.path.exists(scene_path):
            grasp = grasp_scene(args=args)
            grasp.generete_scene(engine, scene_id)
            grasp.gen_mrt(engine, scene_id)








import numpy as np
import os
import pdb
import json

# baxter

# DIR_LIST = [(0, 0, 1)]
DIR_LIST = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]

ROUND_FACTOR = 3
TABLE_COLOR = np.array([1, 0.6, 0])

dataset_path = '/data/cxg8/cleaned_3DVMRD'


def hash_color(color: tuple):
    assert len(color) == 3
    return int(color[0] * 10 ** (ROUND_FACTOR * 2) + color[1] * 10 ** (ROUND_FACTOR * 1) + color[2])


def hash_original_color(original_color: tuple):
    assert len(original_color) == 3
    color = tuple((np.round(np.array(original_color), ROUND_FACTOR) * 10 ** ROUND_FACTOR).astype(np.int))
    return hash_color(color)


def hash_color_array(color_array: np.ndarray):
    assert color_array.shape[1] == 3
    color = (color_array * 10 ** ROUND_FACTOR).astype(int)
    hash_code = color[:, 0] * (10 ** ROUND_FACTOR * 2) + color[:, 1] * 10 ** (ROUND_FACTOR * 1) + color[:, 2]
    return hash_code


def color_array_to_label(color_array: np.ndarray):
    #index = round(color_array[:, 0]*100,8) + round(color_array[:, 1]*10,8) + round(color_array[:, 2]*1, 8)
    index = color_array[:, 0]*100 + color_array[:, 1]*10 + color_array[:, 2]*1
    return index

def read_state_from_json(data_path):
    state_path = os.path.join(data_path, 'final_state.json')
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            result = json.load(f)
        name = []
        path = []
        position = []
        scale = []
        for dict in result:
            name.append(dict['name'])
            path.append(dict['path'])
            pos_str = dict['pos']
            pos = [float(s) for s in pos_str]
            position.append(pos)
    else:
        name, path, position = [],[],[]
        print('final_state.json file in not exist')
    return name, path, position
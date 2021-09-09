# baxter
import pdb
import os
import json
import numpy as np
#zbl

'''
NAME_SCALE = {'002_master_chef_can': [0.4, 0.5],      #102x139
              '003_cracker_box': [0.6, 0.8, 0.9],     #60x158x210
              '004_sugar_box': [0.8, 1.0, 1.2, 1.4],  #39x89x175
              '005_tomato_soup_can': [0.6, 0.8],      #66x101
              '006_mustard_bottle': [0.8, 0.9, 1.0],  #50x85x175
              '007_tuna_fish_can': [0.8, 1.0, 1.2],   #85x33
              '008_pudding_box': [1.0, 1.2, 1.5],     #35x110x89
              '009_gelatin_box': [1.0, 1.2, 1.5, 2.0],#28x85x73
              '010_potted_meat_can': [0.6, 0.8, 1.0], #50x97x82
              '011_banana': [1.0, 1.5],               #36x190   # 1.2 problem RPly: Unable to read magic number from file
              '012_strawberry': [0.9, 1.0, 1.2],      #43.8x55
              '013_apple': [0.5, 0.6, 0.7],           #75
              '014_lemon': [0.6, 0.8, 1.0],           #54x68
              '015_peach': [0.6, 0.8, 1.0],           #59
              '016_pear': [0.6, 0.7, 0.8],            #66.2x100
              '017_orange': [0.5, 0.6, 0.7],          #73
              '018_plum': [0.6, 0.8, 1.0],            #52
              '019_pitcher_base': [0.9, 1.0, 1.2],    #108x235
              '021_bleach_cleanser': [0.6, 0.7, 0.8], #250x98x65
                #'024_bowl': [1.0, 1.2, 1.5],            #159x53
              '025_mug': [1.0],             #80x82  #, 1.2, 1.5
              '026_sponge': [1.0, 1.2, 1.5, 2.0],     #72x114x14
              #'029_plate': [1.0, 1.2],                #258x24
              '030_fork': [1.0, 1.2, 1.5, 2.0],       #14x20x198
              '031_spoon': [1.0, 1.2, 1.5, 2.0],      #14x20x195
              '032_knife': [1.2, 1.5, 2.0],           #14x20x215  #1.0, 
              '033_spatula': [1.0, 1.2, 1.5],         #83x35x350
              '035_power_drill': [1.0, 1.2],     #35x46x184   #, 1.5
              '036_wood_block': [0.5, 0.6],           #85x85x200
              '037_scissors': [1.0, 1.5],          #87x200x14
              '038_padlock': [1.0, 1.2, 1.5],         #24x27x65
              '040_large_marker': [0.8, 1.0, 1.2, 1.5],#18x121
              #'042_adjustable_wrench': [0.9, 1.0, 1.1],  #5x55x205
              #'043_phillips_screwdriver': [1.0, 1.2, 1.5],  #31x215
              '044_flat_screwdriver': [1.0, 1.2, 1.5],#31x215
              '048_hammer': [0.8, 1.0, 1.2],          #34x32x135
              '051_large_clamp': [0.8, 1.0, 1.2],     #165x213x37
              #'053_mini_soccer_ball': [0.2, 0.3, 0.4],#140
              #'054_softball': [0.4, 0.5, 0.6],        #96
              #'055_baseball': [0.5, 0.7],             #80
              '056_tennis_ball': [0.8],           #64.7 ## , 1.0
              '057_racquetball': [0.8, 1.0],           #55.3
              '058_golf_ball': [0.8, 1.0],             #42.7
              '061_foam_brick': [0.8, 0.9, 1.0],       #50x75x50
              '063-a_marbles': [0.8, 1.0, 1.2],
              '065-a_cups': [1.0],
              '065-b_cups': [1.0],
              '065-c_cups': [1.0],
              '065-d_cups': [1.0],
              '065-e_cups': [1.0],
              '065-f_cups': [1.0],
              '065-g_cups': [1.0],
              #'065-h_cups': [1.0], '065-i_cups': [1.0], '065-j_cups': [1.0],
              '071_nine_hole_peg_test': [0.8, 1.0],
              '072-b_toy_airplane': [0.8, 1.0, 1.2],
              '072-c_toy_airplane': [0.8, 1.0, 1.2],
              '072-d_toy_airplane': [0.8, 1.0, 1.2],
              '072-e_toy_airplane': [0.8, 1.0, 1.2],
              '077_rubiks_cube': [0.8, 1.0]            #57x57x57
              }
          
'''
# wang change
def extract_scale(json_path):
    obj_name = []
    scale = []
    #set_scale = []
    #data = np.load(json_path, allow_pickle = True)
    with open(json_path, 'r') as f:
        data = json.load(f)
    for model in data:
        obj_name.append(model['model_name'])
        #set_scale.append(set(model['scale']))
        scale.append([model['obj_id'], model['scale'][0]])
        #scale.append(model['scale'][0])
    #print(scale)
    #print(set_scale)
    scale_model = dict(zip(obj_name,scale))
    #print(scale_model)
    return scale_model
NAME_SCALE = extract_scale('/home/best/data/10/cam1/model_state_list.json')


'''
# formal

NAME_SCALE = {'002_master_chef_can': [0.5, 0.8, 1.0, 1.2],
              '003_cracker_box': [0.5, 0.8, 1.0],
              '004_sugar_box': [0.8, 1.0, 1.2, 1.5],
              '005_tomato_soup_can': [0.5, 0.7, 0.9],
              '006_mustard_bottle': [0.5, 0.8, 1.0, 1.2],
              '007_tuna_fish_can': [0.8, 1.0, 1.2, 1.5],
              '008_pudding_box': [0.8, 1.0, 1.2, 1.5],
              '009_gelatin_box': [0.8, 1.0, 1.2, 1.5, 2.0],
              '010_potted_meat_can': [0.5, 0.8, 1.0],
              '011_banana': [0.8, 1.0, 1.2, 1.5, 2.0],
              '012_strawberry': [0.8, 1.0, 1.2, 1.5, 2.0],
              '013_apple': [0.5, 0.8, 1.0],
              '014_lemon': [0.8, 1.0, 1.2],
              '015_peach': [0.8, 1.0, 1.2],
              '016_pear': [0.5, 0.8, 1.0, 1.2],
              '017_orange': [0.5, 0.8],
              '018_plum': [0.8, 1.0, 1.2],
              '019_pitcher_base': [0.6, 0.8, 1.0],
              '021_bleach_cleanser': [0.8, 1.0],
              '024_bowl': [0.8, 1.0, 1.2],
              '025_mug': [0.8, 1.0, 1.2, 1.5, 2.0],
              '026_sponge': [1.0, 1.2, 1.5, 2.0, 3.0, 4.0],
              '029_plate': [0.8, 1.0, 1.2],
              '033_spatula': [0.8, 1.0, 1.2],
              '035_power_drill': [0.6, 0.8, 1.0, 1.2],
              '036_wood_block': [0.3, 0.5, 0.8],
              '038_padlock': [1.0, 1.2, 1.5, 2],
              '040_large_marker': [0.8, 1.0, 1.2, 1.5],
              '044_flat_screwdriver': [0.8, 1.0, 1.2],
              '048_hammer': [0.8, 1.0, 1.2],
              '051_large_clamp': [0.8, 1.0, 1.2],
              '053_mini_soccer_ball': [0.2, 0.4, 0.7],
              '054_softball': [0.4, 0.6],
              '055_baseball': [0.5, 0.8, 1.0],
              '056_tennis_ball': [0.5, 0.8, 1.0],
              '057_racquetball': [0.8, 1.0, 1.2],
              '058_golf_ball': [0.8, 1.0, 1.2, 1.5],
              '061_foam_brick': [0.8, 1.0, 1.2],
              '063-a_marbles': [0.8, 1.0, 1.2],
              '065-a_cups': [0.8, 1.0],
              '065-b_cups': [0.8, 1.0],
              '065-c_cups': [0.8, 1.0],
              '065-d_cups': [0.8, 1.0],
              '065-e_cups': [0.8, 1.0],
              '065-f_cups': [0.5, 0.8],
              '065-g_cups': [1.0, 1.2],
              '065-h_cups': [0.8, 1.0],
              '065-i_cups': [1.0],
              '065-j_cups': [1.0],
              '071_nine_hole_peg_test': [0.8, 1.0, 1.2],
              '072-b_toy_airplane': [0.8, 1.0, 1.2],
              '072-c_toy_airplane': [0.8, 1.0, 1.2],
              '072-d_toy_airplane': [0.8, 1.0, 1.2],
              '072-e_toy_airplane': [0.8, 1.0, 1.2],
              '077_rubiks_cube': [0.8, 1.0]
              }
'''             
              

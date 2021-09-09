import os
import shutil
import random
import json

def random_pick(some_list, probabilities):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:break
    return item


id2name_path = 'id2name.json'
path = 'new_model_samples'
with open(id2name_path,'r') as f:
    id2name = json.load(f)

def get_sampled_dict(type):
    with open(os.path.join(path, type+'.json'),'r') as f:
        dic = json.load(f)
    return dic


def get_sample_prob(keys):
    prob_dict = {}
    prob_dict = prob_dict.fromkeys(keys)
    importan_objname = ['remote control', 'basket', 'can', 'bowl', 'mug', 'knife', 'bottle', 'cellular telephone', 'car', 'airplane', 'pot', 'clock','sofa','skateboard','laptop']
    factor = 0.4
    import_num=0
    for id in prob_dict.keys():
        if id2name[id] in importan_objname:
            import_num +=1
    if import_num == 0:
        for key,value in prob_dict.items():
            prob_dict[key] = 1/len(keys)
    else:
        importan_factor = factor/import_num
        unimportant_factor = (1-factor)/(len(keys)-import_num)
        for id in prob_dict.keys():
            if id2name[id] in importan_objname:
                prob_dict[id] = importan_factor
            else:
                prob_dict[id] = unimportant_factor
    prob_list = [value for key,value in prob_dict.items()]
    return prob_list

# dic = get_sampled_dict('seen_test')
# prob = get_sample_prob(list(dic))
# print(prob)



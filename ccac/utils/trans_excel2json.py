import pandas as pd
import json
import os
import math
import random

def trans(angle_path):
    df = pd.read_excel(angle_path)
    arguments = {}
    for idx, line in df.iterrows():
        # item_dict = {}
        keys = line.keys()
        argument = None
        for key in keys:
            item = line[key]
            if key == '对比': key = '比较'
            if isinstance(item, float):
                arguments[argument][key] = 1 if item == 1 else 0
            else:
                arguments[item] = {}
                argument = item
    return arguments


def cal_score(name_1, name_2):
    # 根据重复值计算相似度  相似度是name_2相对于name_1计算的
    score = len([n for n in name_2 if n in name_1]) / len(name_2)
    return score


def main(angle_dir):
    filenames = os.listdir(angle_dir)
    filenames = [name for name in filenames if 'xlsx' in name]

    filenames = [name[:-5] for name in filenames]
    grouped_filenames = []
    for name in filenames:
        match_idx = -1
        for i in range(len(grouped_filenames)):
            grp_names = grouped_filenames[i]
            scores = [cal_score(n, name) for n in grp_names if name]
            flag = all([score > score_th for score in scores])
            if flag: 
                match_idx = i
                break
        if match_idx == -1:
            grouped_filenames.append([name])
        else:
            grouped_filenames[match_idx].append(name)
    grp_val_names = random.choices(grouped_filenames, k=math.ceil(len(grouped_filenames)*clip_pct))
    val_names = []
    for grp_names in grp_val_names:
        val_names.extend(grp_names)
    train_names = [name for name in filenames if name not in val_names]
    train_data = {}
    for name in train_names:
        path = angle_dir + name + '.xlsx'
        claim = name
        arguments = trans(path)
        train_data[claim] = arguments
    val_data = {}
    for name in val_names:
        path = angle_dir + name + '.xlsx'
        claim = name
        arguments = trans(path)
        val_data[claim] = arguments
    # val_keys = random.choices(list(data.keys()), k=math.ceil(len(data)*clip_pct))
    # val_data = {k: v for k, v in data.items() if k in val_keys}
    # train_data = {k: v for k, v in data.items() if k not in val_data}
    with open(angle_dir + 'train_data.json', 'w') as f:
        json.dump(train_data, f)
    with open(angle_dir + 'val_data.json', 'w') as f:
        json.dump(val_data, f)


def get_json(angle_dir):
    filenames = os.listdir(angle_dir)
    filenames = [name for name in filenames if 'xlsx' in name]

    filenames = [name[:-5] for name in filenames]
    grouped_filenames = []
    for name in filenames:
        match_idx = -1
        for i in range(len(grouped_filenames)):
            grp_names = grouped_filenames[i]
            scores = [cal_score(n, name) for n in grp_names if name]
            flag = all([score > score_th for score in scores])
            if flag: 
                match_idx = i
                break
        if match_idx == -1:
            grouped_filenames.append([name])
        else:
            grouped_filenames[match_idx].append(name)
    grp_val_names = random.choices(grouped_filenames, k=math.ceil(len(grouped_filenames)*clip_pct))
    # val_names = []
    # for grp_names in grp_val_names:
    #     val_names.extend(grp_names)
    val_names = ['当今中国发展共享经济，更应注重新兴产业创业',
                 '当今中国发展共享经济，更应注重传统产业提升',
                 '新零售是商业模式革命',
                 '新零售不是商业模式革命']
    train_names = [name for name in filenames if name not in val_names]
    train_data = {}
    for name in train_names:
        path = angle_dir + name + '.xlsx'
        claim = name
        arguments = trans(path)
        train_data[claim] = arguments
    val_data = {}
    for name in val_names:
        path = angle_dir + name + '.xlsx'
        claim = name
        arguments = trans(path)
        val_data[claim] = arguments
    # data = {}
    # for name in filenames:
    #     path = angle_dir + name
    #     claim = name[:-5]
    #     arguments = trans(path)
    #     data[claim] = arguments
    # val_keys = random.choices(list(data.keys()), k=math.ceil(len(data)*clip_pct))
    # val_data = {k: v for k, v in data.items() if k in val_keys}
    # train_data = {k: v for k, v in data.items() if k not in val_data}
    with open(angle_dir + 'train_data.json', 'w') as f:
        json.dump(train_data, f)
    with open(angle_dir + 'val_data.json', 'w') as f:
        json.dump(val_data, f)

if __name__ == "__main__":
    clip_pct = 0.1
    score_th = 0.5
    angle_dir = '/root/autodl-tmp/data/ccac/track2/angle/'
    # out_path = '/root/autodl-tmp/data/ccac/track2/angle/angle_data.json'
    get_json(angle_dir)
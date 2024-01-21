import os
import random
import shutil

def cal_score(name_1, name_2):
    # 根据重复值计算相似度  相似度是name_2相对于name_1计算的
    score = len([n for n in name_2 if n in name_1]) / len(name_2)
    return score

def del_file(path_data):
    for i in os.listdir(path_data) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "/" + i#当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)

def clip(original_dir, claim_path, train_dir, test_dir, clip_pct):
    score_th = 0.5
    with open(claim_path) as f:
        filenames = f.readlines()
    filenames = [name[:-1] for name in filenames]
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

    grp_test_names = random.choices(grouped_filenames, k=int(len(grouped_filenames)*clip_pct))
    test_names = []
    for grp_names in grp_test_names:
        test_names.extend(grp_names)
    train_names = [name for name in filenames if name not in test_names]
    train_names = [name+'.txt' for name in train_names]
    test_names = [name+'.txt' for name in test_names]
    del_file(train_dir)
    del_file(test_dir)
    
    for name in train_names:
        shutil.copyfile(original_dir+name, train_dir+name)
    for name in test_names:
        shutil.copyfile(original_dir+name, test_dir+name)
    with open(train_dir+'claims.txt', 'w') as f:
        train_names = [name[:-4]+'\n' for name in train_names]
        f.writelines(train_names)
    with open(test_dir+'claims.txt', 'w') as f:
        test_names = [name[:-4]+'\n' for name in test_names]
        f.writelines(test_names)
    print(123)

def get_claim(original_dir, claim_path):
    filenames = os.listdir(original_dir)
    filenames = [name[:-4]+'\n' for name in filenames]
    # random.shuffle(filenames)
    with open(claim_path, 'w') as f:
        f.writelines(filenames)

if __name__ == "__main__":
    clip_pct = 0.15
    original_dir = '/root/autodl-tmp/data/ccac/track2/original/'
    claim_path = '/root/autodl-tmp/data/ccac/track2/claims.txt'
    train_dir = '/root/autodl-tmp/data/ccac/track2/train/'
    test_dir = '/root/autodl-tmp/data/ccac/track2/test/'
    # get_claim(original_dir, claim_path)
    clip(original_dir, claim_path, train_dir, test_dir, clip_pct)
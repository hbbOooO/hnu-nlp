import os
import json
import random

# 将来自huggingface的imdb数据集转换为json格式的训练集、测试集
train_pos_data_dir = '/root/autodl-tmp/data/imdb/original/aclImdb/train/pos/'
train_neg_data_dir = '/root/autodl-tmp/data/imdb/original/aclImdb/train/neg/'
test_pos_data_dir = '/root/autodl-tmp/data/imdb/original/aclImdb/test/pos/'
test_neg_data_dir = '/root/autodl-tmp/data/imdb/original/aclImdb/test/neg/'

def clean(text):
    text = text.replace('<br>', '')
    text = text.replace('<br />', '')
    return text

train_data = []
test_data = []

train_pos_filenames = os.listdir(train_pos_data_dir)
for name in train_pos_filenames:
    if '.txt' not in name: continue
    with open(train_pos_data_dir + name) as f:
        text = f.read()
    text = clean(text)
    train_data.append({
        'text': text,
        'label': 1
    })
train_neg_filenames = os.listdir(train_neg_data_dir)
for name in train_neg_filenames:
    if '.txt' not in name: continue
    with open(train_neg_data_dir + name) as f:
        text = f.read()
    text = clean(text)
    train_data.append({
        'text': text,
        'label': 0
    })
test_pos_filenames = os.listdir(test_pos_data_dir)
for name in test_pos_filenames:
    if '.txt' not in name: continue
    with open(test_pos_data_dir + name) as f:
        text = f.read()
    text = clean(text)
    test_data.append({
        'text': text,
        'label': 1
    })
test_neg_filenames = os.listdir(test_neg_data_dir)
for name in test_neg_filenames:
    if '.txt' not in name: continue
    with open(test_neg_data_dir + name) as f:
        text = f.read()
    text = clean(text)
    test_data.append({
        'text': text,
        'label': 0
    })

clip_pct = 0.1
train_pos_idx_list = [i for i in range(len(train_data)) if train_data[i]['label']==1]
train_neg_idx_list = [i for i in range(len(train_data)) if train_data[i]['label']==0]
selected_pos_idxes = random.choices(train_pos_idx_list, k=int(clip_pct+len(train_pos_idx_list)))
selected_neg_idxes = random.choices(train_neg_idx_list, k=int(clip_pct*len(train_neg_idx_list)))
selected_train_data = []
val_data = []
for i in range(len(train_data)):
    if i in selected_neg_idxes or i in selected_pos_idxes:
        val_data.append(train_data[i])
    else:
        selected_train_data.append(train_data[i])
train_data = selected_train_data

save_dir = '/root/autodl-tmp/data/imdb/'

with open(save_dir+'train.json', 'w') as f:
    json.dump(train_data, f)

with open(save_dir+'val.json', 'w') as f:
    json.dump(val_data, f)

with open(save_dir+'test.json', 'w') as f:
    json.dump(test_data, f)

print('数据集转化完成')

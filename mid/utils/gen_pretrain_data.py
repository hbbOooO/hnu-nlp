import json
import numpy as np
from tqdm import tqdm


def read_data_from_csv(path_1, path_2):
    data = []
    with open(path_1) as f:
        lines = f.readlines()
    for line in lines:
        line = line[:-1]
        id, description, diagnosis = line.split(',')
        data.append({
            'description': description,
            'diagnosis': diagnosis
        })
    with open(path_2) as f:
        lines = f.readlines()
    for line in lines:
        line = line[:-1]
        id, description = line.split(',')
        data.append({
            'description': description,
            'diagnosis': ''
        })
    return data

def read_data_from_csv_stage2(path_1, path_2, path_3, path_4):
    data = []
    with open(path_1) as f:
        lines = f.readlines()
    for line in lines:
        line = line[:-1]
        id, description, diagnosis = line.split(',')
        data.append({
            'description': description,
            'diagnosis': diagnosis
        })
    with open(path_2) as f:
        lines = f.readlines()
    for line in lines:
        line = line[:-1]
        id, description = line.split(',')
        data.append({
            'description': description,
            'diagnosis': ''
        })
    with open(path_3) as f:
        lines = f.readlines()
    for line in lines:
        line = line[:-1]
        id, description = line.split(',')
        data.append({
            'description': description,
            'diagnosis': ''
        })
    with open(path_4) as f:
        lines = f.readlines()
    for line in lines:
        line = line[:-1]
        id, description, diagnosis, clinical = line.split(',')
        data.append({
            'description': description,
            'diagnosis': diagnosis,
            'clinical': clinical
        })
    with open(path_2) as f:
        lines = f.readlines()
    for line in lines:
        line = line[:-1]
        id, description = line.split(',')
        data.append({
            'description': description,
            'diagnosis': ''
        })
    return data

def get_text(data):
    texts = []
    for item in data:
        description = item['description'].split()
        diagnosis = item['diagnosis'].split()
        description = [description] if isinstance(description, str) else description
        diagnosis = [diagnosis] if isinstance(diagnosis, str) else diagnosis
        texts.append(description)
        if use_label:
            texts.append(diagnosis)
    return texts

def get_text_stage2(data):
    texts = []
    for item in data:
        description = item['description'].split()
        diagnosis = item['diagnosis'].split()
        clinical = item['clinical'].split()
        description = [description] if isinstance(description, str) else description
        diagnosis = [diagnosis] if isinstance(diagnosis, str) else diagnosis
        clinical = [clinical] if isinstance(clinical, str) else clinical
        texts.append(description)
        texts.append(clinical)
        if use_label:
            texts.append(diagnosis)
    return texts

def get_vocab(path):
    with open(path) as f:
        vocabs = f.readlines()
    vocabs = [vocab[:-1] for vocab in vocabs]
    return vocabs
    
def token_masking(texts):
    if not use_token_masking:
        return []
    '''
    与Bert方法一致：
    随机替换句子中15%的单词
    80% 将选中的单词替换成 [MASK]
    10% 将选中的单词替换成 任意单词
    10% 将选中的单词替换成 不变
    PS：在这里，任意单词来自于vocab.txt中的单词
    '''
    pre_texts = []
    # texts = get_text(data)
    # 单词表中，有含义的单词的起始位置和结束位置（含头不含尾）
    start_idx = 105
    end_idx = 1415
    vocabs = get_vocab('/root/autodl-tmp/data/mid/pretrained_bert/vocab.txt')
    min_mask_len = 2
    for text in tqdm(texts):
        rands = np.random.random(len(text))
        pre_text = text.copy()
        if len(pre_text) >= min_mask_len:
            for i in range(len(rands)):
                if rands[i] < 0.15 * 0.8:
                    pre_text[i] = MASK
                elif rands[i] < 0.15 * 0.9:
                    idx = np.random.randint(start_idx, end_idx)
                    pre_text[i] = vocabs[idx]
        pre_texts.append(' '.join(pre_text) + ',' + ' '.join(text) + '\n')
    return pre_texts
            

def sentence_permutation(texts):
    if not use_sentence_permutation:
        return []
    '''
    句子顺序随机打乱
    按照 id=10 为句号 的条件进行句子的划分
    '''
    segment_token = '10'
    pre_texts = []
    for text in tqdm(texts):
        segment_token_ids = [i+1 for i in range(len(text)) if text[i] == segment_token]
        segment_token_ids.insert(0, 0)
        if len(segment_token_ids) in [1, 2]:
            pre_text = text.copy()
            
        else:
            sentences = [text[segment_token_ids[i]:segment_token_ids[i+1]] for i in range(len(segment_token_ids)-1)]
            np.random.shuffle(sentences)
            pre_text = []
            for sentence in sentences:
                pre_text.extend(sentence)
        pre_texts.append(' '.join(pre_text) + ',' + ' '.join(text) + '\n')
    return pre_texts

    


def document_rotation(texts):
    if not use_document_rotation:
        return []
    '''
    均匀地随机选择一个标记，并旋转文档，使其以该标记开始，调整文档的顺序
    '''
    pre_texts = []
    for text in tqdm(texts):
        if len(text) < 3:
            pre_text = text.copy()
        else:
            index = np.random.randint(1, len(text)-1)
            pre_text = text[index:] + text[:index]
            assert len(pre_text) == len(text)
        pre_texts.append(' '.join(pre_text) + ',' + ' '.join(text) + '\n')
    return pre_texts

def token_deletion(texts):
    if not use_token_deletion:
        return []
    '''
    随机删除字符
    文章里面没有提到删除的概率，这里设置成10%
    '''
    pre_texts = []
    min_delete_len = 2
    for text in tqdm(texts):
        rands = np.random.random(len(text))
        pre_text = text.copy()
        if len(pre_text) >= min_delete_len:
            delete_ids = []
            for i in range(len(rands)):
                if rands[i] < 0.1:
                    delete_ids.append(i-len(delete_ids))
            for id in delete_ids:
                pre_text.pop(id)
        pre_texts.append(' '.join(pre_text) + ',' + ' '.join(text) + '\n')
    return pre_texts

def text_infilling(texts):
    if not use_text_infilling:
        return []
    '''
    不定长文本掩码
    掩码文本的长度满足泊松分布
    掩码文本的最大长度为所有文本长度的 0.3
    泊松分布的lambda=3
    '''
    pre_texts = []
    for text in tqdm(texts):
        pre_text = text.copy()
        text_length = len(pre_text)
        max_mask_length = int(len(pre_text) * 0.3)
        masked_length = 0
        # token_masked_list = [False for _ in range(len(text))]
        while masked_length < max_mask_length:
            poisson_length = np.random.poisson(3)
            poisson_length = poisson_length + 1 if poisson_length == 0 else poisson_length 
            start_index = int(np.random.uniform(0, text_length - poisson_length))
            pre_text = pre_text[:start_index] + [MASK] + pre_text[start_index+poisson_length:]
            text_length -= poisson_length - 1
            masked_length += poisson_length
        pre_texts.append(' '.join(pre_text) + ',' + ' '.join(text) + '\n')
    return pre_texts

def stage1():
    path_1 = '/root/autodl-tmp/data/mid/train.csv'
    path_2 = '/root/autodl-tmp/data/mid/preliminary_a_test.csv'
    data = read_data_from_csv(path_1, path_2)

    texts = get_text(data)
    token_masking_texts = token_masking(texts)
    sentence_permutation_texts = sentence_permutation(texts)
    document_rotation_texts = document_rotation(texts)
    token_deletion_texts = token_deletion(texts)
    text_infilling_texts = text_infilling(texts)

    save_dir = '/root/autodl-tmp/data/mid/pretrain_data/'
    train_len = int(len(texts) - 400)
    with open(save_dir + 'token_masking_train.csv', 'w') as f:
        f.writelines(token_masking_texts[:train_len])
    with open(save_dir + 'sentence_permutation_train.csv', 'w') as f:
        f.writelines(sentence_permutation_texts[:train_len])
    with open(save_dir + 'document_rotation_train.csv', 'w') as f:
        f.writelines(document_rotation_texts[:train_len])
    with open(save_dir + 'token_deletion_train.csv', 'w') as f:
        f.writelines(token_deletion_texts[:train_len])
    with open(save_dir + 'text_infilling_train.csv', 'w') as f:
        f.writelines(text_infilling_texts[:train_len])

    with open(save_dir + 'token_masking_val.csv', 'w') as f:
        f.writelines(token_masking_texts[train_len:])
    with open(save_dir + 'sentence_permutation_val.csv', 'w') as f:
        f.writelines(sentence_permutation_texts[train_len:])
    with open(save_dir + 'document_rotation_val.csv', 'w') as f:
        f.writelines(document_rotation_texts[train_len:])
    with open(save_dir + 'token_deletion_val.csv', 'w') as f:
        f.writelines(token_deletion_texts[train_len:])
    with open(save_dir + 'text_infilling_val.csv', 'w') as f:
        f.writelines(text_infilling_texts[train_len:])


def stage2():
    path_1 = '/root/autodl-tmp/data/mid/train.csv'
    path_2 = '/root/autodl-tmp/data/mid/preliminary_a_test.csv'
    path_3 = '/root/autodl-tmp/data/mid/preliminary_b_test.csv'
    path_4 = '/root/autodl-tmp/data/mid/stage2/semi_train.csv'
    data = read_data_from_csv_stage2(path_1, path_2, path_3, path_4)


    texts = get_text(data)
    token_masking_texts = token_masking(texts)
    sentence_permutation_texts = sentence_permutation(texts)
    document_rotation_texts = document_rotation(texts)
    token_deletion_texts = token_deletion(texts)
    text_infilling_texts = text_infilling(texts)

    save_dir = '/root/autodl-tmp/data/mid/stage2/pretrain_data/'
    train_len = int(len(texts) - 400)
    with open(save_dir + 'token_masking_train.csv', 'w') as f:
        f.writelines(token_masking_texts[:train_len])
    with open(save_dir + 'sentence_permutation_train.csv', 'w') as f:
        f.writelines(sentence_permutation_texts[:train_len])
    with open(save_dir + 'document_rotation_train.csv', 'w') as f:
        f.writelines(document_rotation_texts[:train_len])
    with open(save_dir + 'token_deletion_train.csv', 'w') as f:
        f.writelines(token_deletion_texts[:train_len])
    with open(save_dir + 'text_infilling_train.csv', 'w') as f:
        f.writelines(text_infilling_texts[:train_len])

    with open(save_dir + 'token_masking_val.csv', 'w') as f:
        f.writelines(token_masking_texts[train_len:])
    with open(save_dir + 'sentence_permutation_val.csv', 'w') as f:
        f.writelines(sentence_permutation_texts[train_len:])
    with open(save_dir + 'document_rotation_val.csv', 'w') as f:
        f.writelines(document_rotation_texts[train_len:])
    with open(save_dir + 'token_deletion_val.csv', 'w') as f:
        f.writelines(token_deletion_texts[train_len:])
    with open(save_dir + 'text_infilling_val.csv', 'w') as f:
        f.writelines(text_infilling_texts[train_len:])

if __name__ == "__main__":
    # https://github.com/cosmoquester/transformers-bart-pretrain

    

    MASK = '[MASK]'

    use_token_masking = True
    use_sentence_permutation = True
    use_document_rotation = True
    use_token_deletion = True
    use_text_infilling = True

    use_label = False

    stage2()
    
    print('完成')
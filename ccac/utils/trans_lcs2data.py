from jieba import lcut
import random
import json

def trans2vocab(path, vocab_path):
    with open(path) as f:
        lines = f.readlines()
    vocabs = [] 
    for line in lines:
        line = line[:-1]
        claim = line.split(' ')[0]
        words = line.split(' ')[2:]
        words = [word.replace('\'', '').replace('[', '').replace(']', '').replace(',', '') for word in words]
        claim_tokens = lcut(claim)
        vocabs.extend([word for word in words if word not in claim_tokens])
    vocabs = list(set(vocabs))
    vocabs = [vocab+'\n' for vocab in vocabs]
    with open(vocab_path, 'w') as f:
        f.writelines(vocabs)
    print(123)


def trans2data(path, train_data_path, val_data_path):
    with open(path) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line[:-1]
        claim = line.split(' ')[0]
        words = line.split(' ')[2:]
        words = [word.replace('\'', '').replace('[', '').replace(']', '').replace(',', '') for word in words]
        data.append({
            'claim': claim,
            'label': ''.join(words)
        })
    val_data = random.choices(data, k=int(len(data)*clip_pct))
    train_data = [item for item in data if item not in val_data]
    with open(train_data_path, mode='w', encoding='utf-8') as f:
        json.dump(train_data, f)
    with open(val_data_path, mode='w', encoding='utf-8') as f:
        json.dump(val_data_path, f)

if __name__ == "__main__":
    clip_pct = 0.1
    path = '/root/autodl-tmp/data/ccac/track2/lcs/lcs.txt'
    vocab_path = '/root/autodl-tmp/data/ccac/track2/lcs/vocab.txt'
    # trans2vocab(path, vocab_path)
    train_data_path = '/root/autodl-tmp/data/ccac/track2/lcs/train_data.json'
    val_data_path = '/root/autodl-tmp/data/ccac/track2/lcs/val_data.json'
    trans2data(path, train_data_path, val_data_path)


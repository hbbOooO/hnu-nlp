import json
import random
def make_json():
    with open('selection_file.txt', 'r', encoding='utf-8') as train, \
            open('../../data/pretrain_data/target/ljp_10w_target.txt', 'r', encoding='utf-8') as target, \
            open('maked_train.json', 'a', encoding='utf-8') as result:
        sou = train.readlines()
        tar = target.readlines()
        assert len(sou) == len(tar)

        for (i, j) in zip(sou, tar):
            output = {}

            output['source'] = i
            output['target'] = j
            json.dump(output, result, ensure_ascii=False)
            result.write('\n')

def random_file():
    with open('maked_train.json', 'r', encoding='utf-8') as inf, \
            open('maked_train_random.json', 'w', encoding='utf-8') as outf:
        lines = inf.readlines()
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)

        outf.writelines(lines)
random_file()
# make_json()

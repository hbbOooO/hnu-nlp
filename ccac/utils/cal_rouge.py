import os
from rouge import Rouge
from jieba import lcut
import numpy as np


punctuation = ["！", "!", "？", "?", "｡", "。", "＂", "＃", "＄", "％", "＆", "＇", "（", "）",
               "＊", "＋", "∷", "%", "-", '"', "'", "﹥", "<", ">",
               "－", "／", "：", "；", "＜", "＝", "＞", "＠", "［", "＼", "］", "＾", "＿", "｀",
               "｛", "｜", "｝", "～", "｢", "｣", "､", "、", "〃", "》", "「", "」", "『", "』", "【",
               "】", "〔", "〕", "〖", "〗", "〘", "〙", "〚", "〛", "〜", "〝", "〞", "〟", "–",
               "‘", "‛", "”", "„", "‟", "…", "‧", "﹏", ".", "x", "X", "×", "0", "1", "2",
               "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i",
               "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
               "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
               "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "“", ",", "《", "》"]


def cal():
    filenames = os.listdir(data_dir)
    filenames = [name for name in filenames if '.txt' in name]
    scores = []
    rouge = Rouge()
    for name in filenames:
        path = data_dir + name
        claim = name[:-4]
        claim = remove(claim)
        pred = prompt.format(claim)
        with open(path) as f:
            arguments = f.readlines()
        arguments = [argument[:-1] for argument in arguments]
        references_tokens = [lcut(argument) for argument in arguments]
        output_tokens= lcut(pred)                                   
       
        scores.append([claim, rouge.get_scores(
            [' '.join(output_tokens)] * len(references_tokens),
            [' '.join(x) for x in references_tokens], avg=True
        )['rouge-l']['f']])
    rouge_score =np.mean([score[1] for score in scores])
    print(rouge_score)
    for item in scores:
        print(item[0], item[1])

def remove(claim):
    for char in punctuation:
        if char in punctuation:
            claim = claim.replace(char, '')
    return claim


if __name__ == "__main__":
    # 需要修改数据路径和范式
    data_dir = '/root/autodl-tmp/data/ccac/track2/original/'
    prompt = '是的，{}，谢谢。'
    cal()



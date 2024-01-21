"""Functions adding noise to text"""
import jieba
import random
from .utils import *
from tqdm import tqdm

punctuation = ["，", "！", "!", "？", "?", "｡", "。", "＂", "＃", "＄", "％", "＆", "＇", "（", "）",
               "＊", "＋", "，", "∷", "%", "-", '"', "'", "﹥", "<", ">",
               "－", "／", "：", "；", "＜", "＝", "＞", "＠", "［", "＼", "］", "＾", "＿", "｀",
               "｛", "｜", "｝", "～", "｢", "｣", "､", "、", "〃", "》", "「", "」", "『", "』", "【",
               "】", "〔", "〕", "〖", "〗", "〘", "〙", "〚", "〛", "〜", "〝", "〞", "〟", "–",
               "‘", "‛", "”", "„", "‟", "…", "‧", "﹏", ".", "x", "X", "×", "0", "1", "2",
               "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i",
               "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
               "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
               "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "“", ",", "《", "》"]




# 选词错误
def create_nosisy_text(sentence):
    sentence = sentence.strip()
    s_ = list(sentence)
    ids = []
    pro = random.random()  # 获取一个随机数k
    if pro <= 0.8:
        idx = int(random.random() * len(s_))
        idx2 = int(random.random() * len(s_))
        if s_[idx] not in punctuation:
            k = random.random()  # 获取一个随机数k
            if k <= 0.9:
                w2 = homophones_char(s_[idx])  # 从词表中找出w的同音字
                if w2 is not None and w2 != s_[idx]:
                    # print('w2:', w2)
                    # print('idx', idx)
                    # print('s_[idx]：', s_[idx])
                    s_[idx] = w2
                    ids.append(idx)
        if s_[idx2] not in punctuation:
            k = random.random()  # 获取一个随机数k
            if k <= 0.4:
                w2 = homophones_char(s_[idx2])  # 从词表中找出w的同音字
                if w2 is not None and w2 != s_[idx2]:
                    s_[idx2] = w2
                    ids.append(idx2)
        source = ''.join(s_)
        ids = set(ids)
        ids = list(ids)
        return source, sentence, ids
    else:
        # 形近字加噪
        idx = int(random.random() * len(s_))
        if s_[idx] not in punctuation:
            k = random.random()  # 获取一个随机数k
            if k <= 0.9:
                w2 = similar_form_characters(s_[idx])  # 从词表中找出w的形近字
                if w2 is not None and w2 != s_[idx]:
                    s_[idx] = w2
                    ids.append(idx)
        source = ''.join(s_)
        return source, sentence, ids

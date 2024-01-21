"""tools"""
from pypinyin import pinyin, lazy_pinyin, Style
import random


# 同音字
def homophones_char(word):
    with open("csc/utils/noisy_text/ChineseHomophones/chinese_homophone_char.txt", 'r', encoding="utf-8") as f:
        lines = f.readlines()
        # print(lines)
        word_pinyin = lazy_pinyin(word)
        # print(word_pinyin)
        selection = word
        for row in lines:
            str_row = row.split('\n')[0]
            lst_row = str_row.split('\t')
            # print(lst_row)
            if word_pinyin[0] == lst_row[0]:
                lst_row = lst_row[1:40]
                if len(lst_row) > 4:
                    # print(lst_row[0])

                    word_list = lst_row[:len(lst_row)//4+1]

                else:
                    word_list = lst_row
                # print(word_list)
                k = random.choice(word_list)
                selection = k

        return selection


# 形近字
def similar_form_characters(word):
    with open('csc/utils/noisy_text/SimilarCharacter/形近字语料库.txt','r',encoding="utf-8") as f:
        lines = f.readlines()
        lst_raw = []
        for row in lines:
            str_row = row.split('\n')[0]
            lst_row = str_row.split(' ')
            lst_raw.append(lst_row)
        for i in lst_raw:
            if word == i[0]:
                # print('--------------------------------')
                # print('原字为：', word)
                # print('候选词为', i[1])
                a = len(i[1])
                if a == 1:
                    selection = i[1][0]
                elif a == 0:
                    selection = word
                else:
                    k = random.randint(0, a-1)
                    selection = i[1][k]
                # print('更改为：', selection)
                return selection if selection is not None else word

# 同音词
def homophones_word(word):
    with open("./ChineseHomophones/chinese_homophone_word.txt", 'r', encoding="utf-8") as f:
        lines = f.readlines()
        # print(word)
        # os.exit()
        try:
            word_pinyins = lazy_pinyin(word)
            assert len(word_pinyins)==2
            word_pinyin = word_pinyins[0] + '_' + word_pinyins[1]
        except:
            return word
        # print(word_pinyin)
        selection = word
        for row in lines:
            str_row = row.split('\n')[0]
            lst_row = str_row.split('\t')
            # print(lst_row)
            if word_pinyin == lst_row[0]:
                lst_row = lst_row[1:]
                if len(lst_row) != 0:
                    k = random.choice(lst_row)
                    selection = k

        return selection
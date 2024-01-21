# coding:utf-8

import sys
import Levenshtein
import json

src_path = 'selection_file.txt'
tgt_path = '../../data/pretrain_data/target/ljp_10w_target.txt'
# tgt_path = 'cail_pretrain_target.txt'
output_path = 'selection_output.json'

def convert(src_path,tgt_path,output_path):
    with open(src_path, 'r', encoding='utf-8') as f_src, open(tgt_path, 'r', encoding='utf-8') as f_tgt, \
            open(output_path, 'w', encoding='utf-8') as f_out:
        lines_src = f_src.readlines()
        lines_tgt = f_tgt.readlines()
        # lines_sid = f_sid.readlines()
        assert len(lines_src) == len(lines_tgt)
        id_num = 1
        for i in range(len(lines_src)):
            src_line = lines_src[i].strip().replace(' ', '')
            # print(src_line)
            tgt_line = lines_tgt[i].strip().replace(' ', '')
            # sid = eval(lines_sid[i].strip())['pid']
            sid = id_num
            edits = Levenshtein.opcodes(src_line, tgt_line)
            result = {'pid':sid,'target':[]}
            insert_num = 0
            delete_num = 0
            id_num += 1

            for edit in edits:
                if "。" in tgt_line[edit[3]:edit[4]]: # rm 。
                    continue
                if edit[0] == "insert":
                    insert_num = edit[1]
                    # print(edit[1])
                    # print(delete_num)
                    if edit[1] == 0:
                        result['target'].append({"pos": str(edit[1]), "ori": src_line[edit[1]:edit[2] + 1],
                                                 "edit": tgt_line[edit[3]:edit[4] + 1], "type": "miss"})
                    else:
                        if delete_num == edit[1]-3:
                            if delete_num != 0 and result['target'] != []:
                                result['target'].pop()
                                result['target'].append({"pos": str(edit[1] - 3), "ori": src_line[edit[1] - 3:edit[2]],
                                                        "edit": tgt_line[edit[3] - 2:edit[4]], "type": "disorder"})
                            delete_num = 0
                        else:
                            result['target'].append({"pos": str(edit[1]-1), "ori": src_line[edit[1]-1:edit[2]+1],
                                                    "edit": tgt_line[edit[3]-1:edit[4]+1], "type": "miss"})
                elif edit[0] == "replace":
                    result['target'].append(
                        {"pos": str(edit[1]), "ori": src_line[edit[1]:edit[2]], "edit": tgt_line[edit[3]:edit[4]], "type": "char"})
                elif edit[0] == "delete":
                    delete_num = edit[1]
                    if insert_num == edit[1]-1 or insert_num == edit[1]-2:
                        if insert_num != 0:
                            if insert_num == edit[1]-1 and result['target'] != []:
                                result['target'].pop()
                                result['target'].append({"pos": str(edit[1]-1), "ori": src_line[edit[1]-1:edit[2]], "edit": tgt_line[edit[3]-2:edit[4]], "type": "disorder"})
                            elif insert_num == edit[1]-2:
                                if result['target'] != []:
                                    result['target'].pop()
                                result['target'].append({"pos": str(edit[1]-2), "ori": src_line[edit[1]-2:edit[2]],
                                                         "edit": tgt_line[edit[3]-3:edit[4]], "type": "disorder"})
                            insert_num = 0
                    else:
                        result['target'].append({"pos": str(edit[1]), "ori": src_line[edit[1]:edit[2]], "edit": "", "type": "redund"})
            # print(result)
            json.dump(result, f_out, ensure_ascii=False)
            f_out.write('\n')
if __name__ == '__main__':
    convert(src_path, tgt_path, output_path)
import json
import numpy as np
from sklearn.metrics import f1_score
import os
import operator

from common.datasets.base_dataset import BaseDataset
from csc.utils.noisy_text.create_data import create_nosisy_text

class CscDataset(BaseDataset):
    def __init__(self, config, processors):
        super().__init__(config, processors)

    def _read(self):
        data = []
        for path in self.data_paths:
            with open(path, encoding='utf8') as f:
                lines = f.readlines()
            lines = [line[:-1] for line in lines]
            if self.dataset_type == 'train' or self.dataset_type == 'val':
                for line in lines:
                    if line == '': continue
                    original_text, correct_text = line.split('\t')
                    original_text = original_text.replace(' ', '-')
                    correct_text = correct_text.replace(' ', '-')
                    if not self.config.get('random_noise', False):
                        wrong_ids = [i for i in range(len(original_text)) if original_text[i] != correct_text[i]]
                    else:
                        wrong_ids = None
                    data.append({
                        'id': len(data),
                        'original_text': original_text,
                        'wrong_ids': wrong_ids,
                        'correct_text': correct_text
                    })
            else:
                for line in lines:
                    if line == '': continue
                    original_text = line
                    original_text = original_text.replace(' ', '-')
                    data.append({
                        'id': len(data),
                        'original_text': original_text,
                    })
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        if self.config.get('random_noise', False):
            correct_text = item['correct_text']
            noise_text, _, wrong_ids = create_nosisy_text(correct_text)
            item['original_text'] = noise_text
            item['wrong_ids'] = wrong_ids
            # print(item)
        sample = {}
        for processor in self.processors:
            processor(item, sample)
        return sample

    def _convert_gt_label(self):
        data = self.data
        gt_label = {item['id']: item['correct_text'] for item in data}
        self.gt_label = gt_label

    def evaluate(self, prediction):
        gt_label = self.gt_label
        gt_label = {key: gt_label[key] for key in prediction.keys()}
        prediction = {key: prediction[key] for key in prediction.keys()}
        gt_label_sorted = [gt_label[k] for k in sorted(gt_label.keys())]
        prediction_sorted = [prediction[k] for k in sorted(prediction.keys())]

        # det_y_hat = (outputs[2] > 0.5).long()  # det
        # cor_y_hat = torch.argmax((outputs[3]), dim=-1) # word
        # encoded_x = self.tokenizer(cor_text, padding=True, return_tensors='pt')
        # encoded_x.to(self._device)
        # cor_y = encoded_x['input_ids']
        # cor_y_hat *= encoded_x['attention_mask']

        data_dict = {item['id']: item for item in self.data}
        ori_text_sorted = [data_dict[k]['original_text'] for k in sorted(prediction.keys())]

        pred_results = []
        det_acc_results = []
        word_acc_results = []
        pred_text_sorted = []
        tokenizer = self.processors[1].tokenizer
        for ori_text, gt_text, pred_dict in zip(ori_text_sorted, gt_label_sorted, prediction_sorted):
            ori_text_ids = tokenizer(ori_text, add_special_tokens=False)['input_ids']
            gt_text_ids = tokenizer(gt_text, add_special_tokens=False)['input_ids']
            word_pred = pred_dict['word_pred'].tolist()[1:len(ori_text_ids)+1]
            word_res = operator.eq(gt_text_ids, word_pred)
            det_pred = pred_dict['det_pred'].tolist()[1:len(ori_text_ids)+1]
            det_label = pred_dict['det_label'].tolist()[1:len(ori_text_ids)+1]
            det_res = operator.eq(det_pred, det_label)
            pred_results.append(word_pred)
            det_acc_results.append(det_res)
            word_acc_results.append(word_res)
            pred_text = tokenizer.decode(word_pred, skip_special_tokens=False).split(' ')
            if tokenizer.unk_token in pred_text:
                unk_index_list = [i for i in range(len(pred_text)) if pred_text[i] == tokenizer.unk_token]
                for unk_index in unk_index_list:
                    pred_text[unk_index] = ori_text[unk_index]
            # tokenized_ori_text = tokenizer.tokenize(ori_text)
            # mul_index_list = [i for i in range(len(tokenized_ori_text)) if len(tokenized_ori_text[i]) > 1 and tokenized_ori_text[i] != tokenizer.unk_token]
            # for mul_index in mul_index_list:
            #     pred_text[mul_index] = tokenized_ori_text[mul_index]
            pred_text = ''.join(pred_text)
            pred_text_sorted.append(pred_text)

        det_acc = sum(det_acc_results) / len(det_acc_results)
        word_acc = sum(word_acc_results) / len(word_acc_results)

        sentence_pairs = [[ori_text, gt_text, pred_text] for ori_text, gt_text, pred_text in 
            zip(ori_text_sorted, gt_label_sorted, pred_text_sorted) if len(ori_text) == len(gt_text) == len(pred_text)]
        
        sentence_fpr = self._sentence_fpr(sentence_pairs)
        char_detect_precision, char_detect_recall, char_detect_f1 = self._char_detect(sentence_pairs)
        char_correct_precision, char_correct_recall, char_correct_f1 = self._char_correct(sentence_pairs)

        metric = {
            'detect_accurancy': det_acc,
            'correct_accurancy': word_acc,
            'sentence_fpr': sentence_fpr,
            'detect_precision': char_detect_precision,
            'detect_recall': char_detect_recall,
            'detect_f1': char_detect_f1,
            'correct_precision': char_correct_precision,
            'correct_recall': char_correct_recall,
            'correct_f1': char_correct_f1
        }

        # print(123)
        # metric = {
        #     'cider_score': cider_score,
        #     'rouge_1': result['rouge-1']['f'],
        #     'rouge_2': result['rouge-2']['f'],
        #     'rouge_L': result['rouge-l']['f'],
        # }

        return metric

    def _sentence_fpr(self, sentence_pairs):
        all_num = 0
        cuo = 0
        for sentence in sentence_pairs:
            if sentence[0] == sentence[1]:
                all_num += 1
                if sentence[0] != sentence[2]:
                    cuo += 1
        if all_num == 0:
            fpr = -1
        else:
            fpr = cuo/all_num*100
        # print(f"sentence_FPR：{fpr}")
        return fpr

    def _char_detect(self, sentence_pairs):
        total = 0  # 总的句子数
        cor = 0  # 正确识别错误的句子数
        al_wro = 0  # 算法识别为“有错误”的句子数
        wro = 0  # 语料中所有错误句子数
        k = 0
        for sentence in sentence_pairs:
            k += 1
            total += 1
            lines0 = sentence[0]
            lines1 = sentence[1]
            lines2 = sentence[2]
            # if len(lines0) != len(lines1) or len(lines1) != len(lines2):
            #     print(f"文本长度不一致")
            #     print(sentence)
            assert len(lines0) == len(lines1) and len(lines1) == len(lines2)
            lines_list = [[lines0[i], lines1[i], lines2[i]] for i in range(min(len(lines0), len(lines1), len(lines2)))]
            for char_list in lines_list:
                if char_list[0] != char_list[1] and char_list[0] != char_list[2]:
                    cor += 1
                if char_list[0] != char_list[1]:
                    wro += 1
                if char_list[0] != char_list[2]:
                    al_wro += 1

        try:
            precision = (cor / al_wro) * 100
        except:
            precision = -1
        try:
            recall = (cor / wro) * 100
        except:
            recall = -1
        try:
            f1 = precision * recall * 2 / (precision + recall)
        except:
            f1 = -1

        # char_detect_precision = precision
        # char_detect_recall = recall
        # char_detect_f1 = f1
        # print(char_detect_precision, char_detect_recall, char_detect_f1)
        # print(f"char_detect_precision：{precision}({cor}/{al_wro})")
        # print(f"char_detect_recall：{recall}({cor}/{wro})")
        # print(f"char_detect_F1：{f1}")
        return precision, recall, f1


    def _char_correct(self, sentence_pairs):
        total = 0  # 总的句子数
        TP = 0  # 正确识别错误的句子数
        FP = 0  # 非错别字被误报为错别字
        FN = 0  # 错别字未能正确识别错别字
        k = 0
        for sentence in sentence_pairs:
            k += 1
            total += 1
            lines0 = sentence[0]
            lines1 = sentence[1]
            lines2 = sentence[2]
            lines_list = [[lines0[i], lines1[i], lines2[i]] for i in range(min(len(lines0), len(lines1), len(lines2)))]
            for char_list in lines_list:
                if char_list[0] != char_list[1] and char_list[1] == char_list[2]:
                    TP += 1
                if char_list[2] != char_list[1] and char_list[0] != char_list[2]:
                    FP += 1
                if char_list[0] != char_list[1] and char_list[1] != char_list[2]:
                    FN += 1

        al_wro = TP + FP
        wro = TP + FN
        cor = TP
        try:
            precision = (cor / al_wro) * 100
        except:
            precision = -1
        try:
            recall = (cor / wro) * 100
        except:
            recall = -1
        try:
            f1 = precision * recall * 2 / (precision + recall)
        except:
            f1 = -1
        # print(f"char_correct_precision：{precision}({cor}/{al_wro})")
        # print(f"char_correct_recall：{recall}({cor}/{wro})")
        # print(f"char_correct_F1：{f1}")
        return precision, recall, f1
        

    def out(self, prediction, path):
        dir = path[:-len(path.split('/')[-1])]
        if not os.path.exists(dir): os.makedirs(dir)

        prediction_sorted = [prediction[k] for k in sorted(prediction.keys())]
        data_dict = {item['id']: item for item in self.data}
        ori_text_sorted = [data_dict[k]['original_text'] for k in sorted(prediction.keys())]

        pred_text_sorted = []
        tokenizer = self.processors[1].tokenizer
        for ori_text, pred_dict in zip(ori_text_sorted, prediction_sorted):
            ori_text_ids = tokenizer(ori_text, add_special_tokens=False)['input_ids']
            word_pred = pred_dict['word_pred'].tolist()[1:len(ori_text_ids)+1]
            pred_text = tokenizer.decode(word_pred, skip_special_tokens=False).split(' ')
            if tokenizer.unk_token in pred_text:
                unk_index_list = [i for i in range(len(pred_text)) if pred_text[i] == tokenizer.unk_token]
                for unk_index in unk_index_list:
                    pred_text[unk_index] = ori_text[unk_index]
            pred_text = ''.join(pred_text)
            pred_text_sorted.append(pred_text)

        out_data = [
            ori_text + '\t' + pred_text + '\n' for ori_text, pred_text in zip(ori_text_sorted, pred_text_sorted)
        ]

        
        with open(path, mode='w', encoding='utf-8') as f:
            f.writelines(out_data)
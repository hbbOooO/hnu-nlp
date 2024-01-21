"""
日期: 2023年6月25日
作者: 韩明杰
数据来源: https://ai.stanford.edu/~amaas/data/sentiment/
"""
import torch
from torch import nn
from common.datasets.base_dataset import BaseDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

class IMDBDataset(BaseDataset):
    def __init__(self, config, processors):
        super().__init__(config, processors)

    def _read(self):
        data = []
        for path in self.data_paths:
            with open(path, encoding='utf8') as f:
                lines = json.load(f)
            for line in lines:
                item = {}
                item['text'] = line['text']
                if self.dataset_type == 'train' or self.dataset_type == 'val':
                    item['labels'] = line['label']
                data.append(item)
        self.data = data

    def _convert_gt_label(self):
        data = self.data
        gt_label = {item['id']: item['labels'] for item in data}
        self.gt_label = gt_label
    

    def evaluate(self, prediction):
        gt_label = self.gt_label
        gt_label = {key: gt_label[key]['pred'] for key in prediction.keys()}
        # sort
        # sorted_keys = sorted(gt_label.keys())
        gt_label_sorted = [gt_label[k] for k in sorted(gt_label.keys())]
        prediction_sorted = [prediction[k][0] for k in sorted(prediction.keys())]

        accuracy = sum([prediction_item == gt_label_item for prediction_item, gt_label_item in \
            zip(prediction_sorted, gt_label_sorted)]) / len(prediction_sorted)

        metric = {
            'acc': accuracy,
        }

        return metric
    
    def out(self, prediction, path):
        dir = path[:-len(path.split('/')[-1])]
        if not os.path.exists(dir): os.makedirs(dir)

        # def get_text(label, sentences):
        #     inds = np.where(label)[0]
        #     text = ''.join([sentences[ind] for ind in inds])
        #     return text

        text_sentences_dict = {item['id']: item['text_sentences'] for item in self.data}
        text_id_dict = {item['id']: item['text_id'] for item in self.data}
        # pred_texts = [get_text(prediction[k], text_sentences_dict[k]) for k in sorted(prediction.keys())]

        prediction = [{"id": text_id_dict[k], "summary": self.get_text(prediction[k], text_sentences_dict[k])} for k in sorted(prediction.keys())]
        for item in prediction:
            if item['summary'] == "":
                id = [k for k, v in text_id_dict.items() if v == item['id']][0]
                item['summary'] = ''.join(text_sentences_dict[id][:3])
        prediction = [json.dumps(item,ensure_ascii=False)+"\n" for item in prediction]
        with open(path, mode='w', encoding='utf8') as f:
            f.writelines(prediction)
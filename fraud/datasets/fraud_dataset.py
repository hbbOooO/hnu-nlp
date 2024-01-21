import torch
from torch import nn
# from torch.utils.data.dataset import Dataset
from common.datasets.base_dataset import BaseDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

# from .processors.processors import Pipeline

class FraudDataset(BaseDataset):

    LABELS = ['刷单返利类', '冒充电商物流客服类', '虚假网络投资理财类', '贷款、代办信用卡类',
              '虚假征信类', '虚假购物、服务类', '冒充公检法及政府机关类', '冒充领导、熟人类',
              '网络游戏产品虚假交易类', '网络婚恋、交友类（非虚假网络投资理财类）', '冒充军警购物类', '网黑案件']

    def __init__(self, config, processors):
        super().__init__(config, processors)


    # read data
    def _read(self):
        data = []
        for path in self.data_paths:
            with open(path, encoding='utf8') as f:
                lines = f.readlines()
            lines = [json.loads(line[:-1]) for line in lines]
            if self.dataset_type == 'train' or self.dataset_type == 'val':
                for line in lines:
                    data.append({
                        'id': len(data),
                        'case_id': line['案件编号'],
                        'text': line['案情描述'],
                        'label': self.LABELS.index(line['案件类别'])
                    })
            else:
                for line in lines:
                    data.append({
                        'id': len(data),
                        'case_id': line['案件编号'],
                        'text': line['案情描述'],
                    })
        self.data = data

    def _convert_gt_label(self):
        data = self.data
        gt_label = {item['id']: item['label'] for item in data}
        self.gt_label = gt_label
    

    def evaluate(self, prediction):
        # prediction = self.get_new_prediction(prediction, clear)
        gt_label = self.gt_label
        gt_label = {key: gt_label[key] for key in prediction.keys()}
        # sort
        # sorted_keys = sorted(gt_label.keys())
        gt_label_sorted = [gt_label[k] for k in sorted(gt_label.keys())]
        prediction_sorted = [prediction[k][0] for k in sorted(prediction.keys())]

        macro_f1 = f1_score(gt_label_sorted, prediction_sorted, average='macro')

        metric = {
            'macro_f1': macro_f1,
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
import torch
import json
import numpy as np

from tqdm import tqdm
from rouge import Rouge
from collections import defaultdict
from jieba import lcut
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from common.datasets.base_dataset import BaseDataset
from common.logger import Logger

class CcacDataset(BaseDataset):
    # ARGUMENT_CLASS = ['定义', '影响', '方法', '背景', '反驳', '举例', '比较', '未分类']

    def __init__(self, config, processors):
        # super(CcacDataset, self).__init__(config, processors)
        super().__init__(config, processors)
        

    def _read(self):
        data = []
        for path in self.data_paths:
            with open(path, encoding='utf-8') as f:
                lines = json.load(f)
            if self.dataset_type == 'train' or self.dataset_type == 'val':
                for line in lines:
                    data.append({
                        'id': len(data),
                        'claim': line['claim'],
                        'label': line['label']
                    })
            else:
                for line in lines:
                    data.append({
                        'id': len(data),
                        'claim': line['claim']
                    })
        self.data = data

    def _convert_gt_label(self):
        data = self.data
        gt_label = {item['id']: item['label'] for item in data}
        self.gt_label = gt_label
        

    def evaluate(self, prediction):
        gt_label = self.gt_label
        gt_label = {key: gt_label[key] for key in prediction.keys()}
        prediction = {key: [prediction[key]['pred'].tolist()]
                      for key in prediction.keys()}
        # sort
        # sorted_keys = sorted(gt_label.keys()) c
        gt_label_sorted = [gt_label[k] for k in sorted(gt_label.keys())]
        prediction_sorted = [prediction[k] for k in sorted(prediction.keys())]

        # bleu = []
        scores = defaultdict(list)
        smoothing = SmoothingFunction()
        rouge = Rouge()
        tokenizer = self.processors[1]
        # Logger.get_logger().info('Start to calculate the metric. The process is slow. Please wait patiently.')
        for gt_label_list, prediction_list in zip(gt_label_sorted, prediction_sorted):
            # gt_label_list = gt_label_sorted[i]
            # prediction_list = prediction_sorted[i]
            references_tokens = [lcut(x.strip()) for x in gt_label_list]
            for prediction_item in  prediction_list:
                decoded_pred = tokenizer.tokenizer.decode(prediction_item, skip_special_tokens=True)
                output_tokens = lcut(decoded_pred)
        
                scores['bleu'].append(sentence_bleu(references_tokens, output_tokens,
                                                        smoothing_function=smoothing.method1))

                for key, result in rouge.get_scores(
                    [' '.join(output_tokens)] * len(references_tokens),
                    [' '.join(x) for x in references_tokens], avg=True
                ).items():
                    # key: str
                    # result: Dict[str, float]
                    scores[key].append(result['f'])
        scores = {key: np.mean(values) for key, values in scores.items()}
        metric = {
            'bleu': scores['bleu'],
            'rouge_1': scores['rouge-1'],
            'rouge_2': scores['rouge-2'],
            'rouge_L': scores['rouge-l'],
        }
        return metric
    






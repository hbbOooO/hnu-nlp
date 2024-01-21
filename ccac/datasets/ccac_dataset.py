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
    ARGUMENT_CLASS = ['定义', '影响', '方法', '背景', '反驳', '举例', '比较', '未分类']

    def __init__(self, config, processors):
        # super(CcacDataset, self).__init__(config, processors)
        super(CcacDataset, self).__init__(config, processors)
        

    def _read(self):
        if self.dataset_type == 'train' or self.dataset_type == 'val':
            self.use_argument_cls = self.config['use_argument_cls']
        data = []
        for path in self.data_paths:
            if self.dataset_type == 'train':
                if self.use_argument_cls:
                    # 读取打完标签的数据
                    # 读取文件类型 json
                    with open(path, encoding='utf8') as f:
                        lines = json.load(f)
                    for claim, claim_value in lines.items():
                        for argument, arguument_value in claim_value.items():
                            if arguument_value[self.ARGUMENT_CLASS[-1]] == 1: continue
                            int_cls = ''.join([str(arguument_value[k]) for k in self.ARGUMENT_CLASS if k in arguument_value.keys()])
                            item = {
                                'id': len(data),
                                'claim': claim,
                                'int_cls': int_cls,
                                'argument': argument,
                                'argument_cls': self.ARGUMENT_CLASS,
                                'use_argument_cls': True
                            }
                            data.append(item)
                else:
                    # 读取原始数据
                    # 读取文件类型  claims.txt
                    with open(path, encoding='utf8') as f:
                        lines = f.readlines()
                    lines = [line[:-1] for line in lines]
                    data_root_dir = self.config['data_root_dir']
                    for claim in lines:
                        with open(data_root_dir + claim + '.txt') as f:
                            arguments = f.readlines()
                        for argument in arguments:
                            argument = argument[:-1]
                            item = {
                                'id': len(data),
                                'claim': claim,
                                'argument': argument,
                                'use_argument_cls': False
                            }
                            data.append(item)
            elif self.dataset_type == 'val':
                if self.use_argument_cls:
                    # 读取打完标签的数据
                    # 读取文件类型 json
                    with open(path, encoding='utf8') as f:
                        lines = json.load(f)
                    for claim, claim_value in lines.items():
                        arguments = [k for k, v in claim_value.items()]
                        item = {
                            'id': len(data),
                            'claim': claim,
                            'arguments': arguments,
                            'use_argument_cls': True
                        }
                        data.append(item)
                else:
                    # 读取原始数据
                    # 读取文件类型  claims.txt
                    with open(path, encoding='utf8') as f:
                        lines = f.readlines()
                    lines = [line[:-1] for line in lines]
                    data_root_dir = self.config['data_root_dir']
                    for claim in lines:
                        with open(data_root_dir + claim + '.txt') as f:
                            arguments = f.readlines()
                        arguments = [argument[:-1] for argument in arguments]
                        item = {
                            'id': len(data),
                            'claim': claim,
                            'arguments': arguments,
                            'use_argument_cls': False
                        }
                        data.append(item)
            else:
                with open(path, encoding='utf-8') as f:
                    claims = f.readlines()
                claims = [claim[:-1] for claim in claims]
                data.extend([{
                    'id': i,
                    'claim': claims[i]
                } for i in range(len(claims))])
        self.data = data

    def _convert_gt_label(self):
        data = self.data
        if self.dataset_type == 'val':
            argument_by_id = {item['id']: item['arguments'] for item in data}
            self.gt_label = argument_by_id
        else:
            argument_by_cliam = {}
            argument_by_id = {}
            for item in data:
                if item['claim'] not in argument_by_cliam:
                    argument_by_cliam[item['claim']] = [item['argument']]
                else:
                    argument_by_cliam[item['claim']].append(item['argument'])
                # claim_by_id[item['id']] = item['claim']
            for item in data:
                argument_by_id[item['id']] = argument_by_cliam[item['claim']]
            # argument_by_id = {item['id']: item['arguments'] for item in data}
            self.gt_label = argument_by_id
            # self.claim_by_id = claim_by_id

    
    def evaluate(self, prediction):
        gt_label = self.gt_label
        gt_label = {key: gt_label[key] for key in prediction.keys()}
        prediction = {key: 
                      [
                          prediction[key]['pred_0'].tolist(),
                          prediction[key]['pred_1'].tolist(),
                          prediction[key]['pred_2'].tolist(),
                          prediction[key]['pred_3'].tolist(),
                          prediction[key]['pred_4'].tolist(),
                          ]
                      for key in prediction.keys()}
        # sort
        # sorted_keys = sorted(gt_label.keys()) c
        gt_label_sorted = [gt_label[k] for k in sorted(gt_label.keys())]
        prediction_sorted = [prediction[k] for k in sorted(prediction.keys())]

        # bleu = []
        scores = defaultdict(list)
        smoothing = SmoothingFunction()
        rouge = Rouge()
        tokenizer = self.processors['TextProcessor']
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
    






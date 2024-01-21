import json
import numpy as np
import lawrouge
import os

from common.datasets.base_dataset import BaseDataset
from mid.utils.ciderd import CiderD
from mid.utils.bleu import Bleu

class MidDataset(BaseDataset):
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
                    text_id, text, label = line.split(',')
                    data.append({
                        'id': len(data),
                        'text': text,
                        'text_id': text_id,
                        'label': label
                    })
            else:
                for line in lines:
                    text_id, text = line.split(',')
                    data.append({
                        'id': len(data),
                        'text': line['text'],
                        'text_id': line['id'],
                    })
        self.data = data

    def _convert_gt_label(self):
        data = self.data
        gt_label = {item['id']: item['label'] for item in data}
        self.gt_label = gt_label

    def evaluate(self, prediction):
        gt_label = self.gt_label
        gt_label = {key: gt_label[key] for key in prediction.keys()}
        prediction = {key: prediction[key]['pred'].tolist() for key in prediction.keys()}
        # sort
        # sorted_keys = sorted(gt_label.keys())
        gt_label_sorted = [gt_label[k] for k in sorted(gt_label.keys())]
        prediction_sorted = [prediction[k] for k in sorted(prediction.keys())]
        # metrics = Smoother(100)
        res, gts = [], {}
        predictions, labels = prediction_sorted, gt_label_sorted
        mid_processor = self.processors[1]
        decoded_preds = mid_processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        # labels = np.where(labels != -100, labels, mid_processor.tokenizer.pad_token_id)
        # decoded_labels = mid_processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = labels

        tot = 0
        for i in range(len(decoded_preds)):
            res.append({'image_id': tot, 'caption': [decoded_preds[i]]})
            gts[tot] = [decoded_labels[i]]
            tot += 1
        CiderD_scorer = CiderD(df='corpus', sigma=15)
        cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
        # metrics.update(cider=cider_score)
        # print(metrics.value())

        # rouge = lawrouge.Rouge()
        # result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)

        # 保存模型
        # trainer.save_model('/root/autodl-tmp/save/mid/results{}'.format(cider_score))
        metric = {
            'cider_score': cider_score,
            # 'rouge_1': result['rouge-1']['f'],
            # 'rouge_2': result['rouge-2']['f'],
            # 'rouge_L': result['rouge-l']['f'],
        }

        # result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}

        # result = {key: value * 100 for key, value in result.items()}

        # metric = {
        #     'acc': accuracy,
        # }

        return metric

    def out(self, prediction, path):
        dir = path[:-len(path.split('/')[-1])]
        if not os.path.exists(dir): os.makedirs(dir)

        # self.max_out_length = self.config['max_out_length']
        text_id_dict = {item['id']: item['text_id'] for item in self.data}
        mid_processor = self.processors[1]
        lines = []
        # prediction_sorted = [prediction[k] for k in sorted(prediction.keys())]
        for k in sorted(prediction.keys()):
            pred = prediction[k]
            text_id = text_id_dict[k]
            decoded_pred = mid_processor.tokenizer.decode(pred, skip_special_tokens=True)
            lines.append(text_id + ',' + decoded_pred + '\n')
        with open(path, mode='w', encoding='utf-8') as f:
            f.writelines(lines)

class MidStage2Dataset(BaseDataset):
    def __init__(self, config, processors):
        super().__init__(config, processors)

    def _read(self):
        data = []
        for path in self.data_paths:
            with open(path, encoding='utf8') as f:
                lines = f.readlines()
            lines = [line[:-1] for line in lines]
            # lines = [json.loads(line) for line in lines]
            if self.dataset_type == 'train' or self.dataset_type == 'val':
                for line in lines:
                    item = line.split(',')
                    if len(item) == 3: 
                        text_id, text, summary = item
                        clinical = ''
                    elif len(item) == 4:
                        text_id, text, summary, clinical = item
                    data.append({
                        'id': len(data),
                        'text': text,
                        'text_id': text_id,
                        'label': summary,
                        'clinical': clinical
                    })
            else:
                for line in lines:
                    item = line.split(',')
                    if len(item) == 2: 
                        text_id, text = item
                        clinical = ''
                    elif len(item) == 3:
                        text_id, text, clinical = item
                    data.append({
                        'id': len(data),
                        'text': text,
                        'text_id': text_id,
                        'clinical': clinical
                    })
        self.data = data

    def _convert_gt_label(self):
        data = self.data
        gt_label = {item['id']: item['label'] for item in data}
        self.gt_label = gt_label

    def evaluate(self, prediction):
        gt_label = self.gt_label
        gt_label = {key: gt_label[key] for key in prediction.keys()}
        prediction = {key: prediction[key]['pred'].tolist() for key in prediction.keys()}
        # sort
        # sorted_keys = sorted(gt_label.keys())
        gt_label_sorted = [gt_label[k] for k in sorted(gt_label.keys())]
        prediction_sorted = [prediction[k] for k in sorted(prediction.keys())]
        # metrics = Smoother(100)
        res, gts = [], {}
        predictions, labels = prediction_sorted, gt_label_sorted
        mid_processor = self.processors[1]
        decoded_preds = mid_processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        # labels = np.where(labels != -100, labels, mid_processor.tokenizer.pad_token_id)
        # decoded_labels = mid_processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = labels

        tot = 0
        for i in range(len(decoded_preds)):
            res.append({'image_id': tot, 'caption': [decoded_preds[i]]})
            gts[tot] = [decoded_labels[i]]
            tot += 1
        CiderD_scorer = CiderD(df='corpus', sigma=15)
        cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)

        blue_res = {item['image_id']: item['caption'] for item in res}
        blue_gts = {k: gts[k] for k, v in blue_res.items()}
        blue_scorer = Bleu()
        blue_score = blue_scorer.compute_score(blue_gts, blue_res)

        # rouge = lawrouge.Rouge()
        # result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)

        # 保存模型
        # trainer.save_model('/root/autodl-tmp/save/mid/results{}'.format(cider_score))
        metric = {
            'cider_score': cider_score,
            'blue_score': blue_score[0][3],
        }

        # result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}

        # result = {key: value * 100 for key, value in result.items()}

        # metric = {
        #     'acc': accuracy,
        # }

        return metric

        # return result

    def out(self, prediction, path):
        dir = path[:-len(path.split('/')[-1])]
        if not os.path.exists(dir): os.makedirs(dir)

        # self.max_out_length = self.config['max_out_length']
        text_id_dict = {item['id']: item['text_id'] for item in self.data}
        mid_processor = self.processors[1]
        lines = []
        # prediction_sorted = [prediction[k] for k in sorted(prediction.keys())]
        for k in sorted(prediction.keys()):
            pred = prediction[k]['pred']
            text_id = text_id_dict[k]
            decoded_pred = mid_processor.tokenizer.decode(pred, skip_special_tokens=True)
            lines.append(text_id + ',' + decoded_pred + '\n')
        with open(path, mode='w', encoding='utf-8') as f:
            f.writelines(lines)

        

import json
import lawrouge

from common.datasets.base_dataset import BaseDataset


class PretrainDataset(BaseDataset):
    def __init__(self, config, processors):
        super().__init__(config, processors)

    def _read(self):
        data = []
        for path in self.data_paths:
            with open(path, encoding='utf8') as f:
                lines = f.readlines()
            lines = [json.loads(line) for line in lines]
            if self.dataset_type == 'train' or self.dataset_type == 'val':
                for line in lines:
                    data.append({
                        'id': len(data),
                        'text': line['pretrained_text'],
                        'label': line['label']
                    })
            else:
                for line in lines:
                    data.append({
                        'id': len(data),
                        'text': line['pretrained_text'],
                    })
        self.data = data

    def _convert_gt_label(self):
        data = self.data
        gt_label = {item['id']: item['label'] for item in data}
        self.gt_label = gt_label

    def evaluate(self, prediction):
        gt_label = self.gt_label
        gt_label = {key: gt_label[key] for key in prediction.keys()}
        prediction = {key: prediction[key].tolist() for key in prediction.keys()}
        # sort
        # sorted_keys = sorted(gt_label.keys())
        gt_label_sorted = [gt_label[k] for k in sorted(gt_label.keys())]
        prediction_sorted = [prediction[k] for k in sorted(prediction.keys())]
        # metrics = Smoother(100)
        res, gts = [], {}
        predictions, labels = prediction_sorted, gt_label_sorted
        mid_processor = self.processors[1]
        decoded_preds = mid_processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        text_dict = {item['id']: item['text'] for item in self.data}
        text_sorted = [text_dict[k] for k in sorted(prediction.keys())]
        for i in range(len(decoded_preds)):
            if decoded_preds[i] == '':
                decoded_preds[i] = text_sorted[i]
        # Replace -100 in the labels as we can't decode them.
        # labels = np.where(labels != -100, labels, mid_processor.tokenizer.pad_token_id)
        # decoded_labels = mid_processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = labels

        # tot = 0
        # for i in range(len(decoded_preds)):
        #     res.append({'image_id': tot, 'caption': [decoded_preds[i]]})
        #     gts[tot] = [decoded_labels[i]]
        #     tot += 1
        # CiderD_scorer = CiderD(df='corpus', sigma=15)
        # cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
        # metrics.update(cider=cider_score)
        # print(metrics.value())

        rouge = lawrouge.Rouge()
        result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
        
        metric = {
            # 'cider_score': cider_score,
            'rouge_1': result['rouge-1']['f'],
            'rouge_2': result['rouge-2']['f'],
            'rouge_L': result['rouge-l']['f'],
        }

        return metric
    
    def out(self, prediction, path):
        pass



class MLMDataset(BaseDataset):
    def __init__(self, config, processors):
        super().__init__(config, processors)

    def _read(self):
        data = []
        for path in self.data_paths:
            with open(path, encoding='utf8') as f:
                lines = f.readlines()
            lines = [json.loads(line) for line in lines]
            if self.dataset_type == 'train' or self.dataset_type == 'val':
                for line in lines:
                    data.append({
                        'id': len(data),
                        'text': line['text'],
                        'text_id': line['id'],
                        'label': line['summary']
                    })
            else:
                for line in lines:
                    data.append({
                        'id': len(data),
                        'text': line['text'],
                        'text_id': line['id'],
                    })
        self.data = data

    def _convert_gt_label(self):
        data = self.data
        gt_label = {item['id']: item['text'] for item in data}
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

        text_dict = {item['id']: item['text'] for item in self.data}
        text_sorted = [text_dict[k] for k in sorted(prediction.keys())]
        for i in range(len(decoded_preds)):
            if decoded_preds[i] == '':
                decoded_preds[i] = text_sorted[i]
        # Replace -100 in the labels as we can't decode them.
        # labels = np.where(labels != -100, labels, mid_processor.tokenizer.pad_token_id)
        # decoded_labels = mid_processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = labels

        # tot = 0
        # for i in range(len(decoded_preds)):
        #     res.append({'image_id': tot, 'caption': [decoded_preds[i]]})
        #     gts[tot] = [decoded_labels[i]]
        #     tot += 1
        # CiderD_scorer = CiderD(df='corpus', sigma=15)
        # cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
        # metrics.update(cider=cider_score)
        # print(metrics.value())

        rouge = lawrouge.Rouge()
        result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
        
        metric = {
            # 'cider_score': cider_score,
            'rouge_1': result['rouge-1']['f'],
            'rouge_2': result['rouge-2']['f'],
            'rouge_L': result['rouge-l']['f'],
        }

        return metric
    
    def out(self, prediction, path):
        pass


class PretrainStage2Dataset(BaseDataset):
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
                    pretrained_text, label = line.split(',')
                    data.append({
                        'id': len(data),
                        'text': pretrained_text,
                        'label': label
                    })
            else:
                for line in lines:
                    pretrained_text, label = line.split(',')
                    data.append({
                        'id': len(data),
                        'text': pretrained_text,
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

        # tot = 0
        # for i in range(len(decoded_preds)):
        #     res.append({'image_id': tot, 'caption': [decoded_preds[i]]})
        #     gts[tot] = [decoded_labels[i]]
        #     tot += 1
        # CiderD_scorer = CiderD(df='corpus', sigma=15)
        # cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
        # metrics.update(cider=cider_score)
        # print(metrics.value())

        rouge = lawrouge.Rouge()
        result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
        
        metric = {
            # 'cider_score': cider_score,
            'rouge_1': result['rouge-1']['f'],
            'rouge_2': result['rouge-2']['f'],
            'rouge_L': result['rouge-l']['f'],
        }

        return metric
    
    def out(self, prediction, path):
        pass



class MLMStage2Dataset(BaseDataset):
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
                    elif len(item) == 4:
                        text_id, text, summary, _ = item
                    data.append({
                        'id': len(data),
                        'text': text,
                        'text_id': text_id,
                        'label': summary
                    })
            else:
                for line in lines:
                    item = line.split(',')
                    if len(item) == 3: 
                        text_id, text, summary = item
                    elif len(item) == 4:
                        text_id, text, summary, _ = item
                    data.append({
                        'id': len(data),
                        'text': text,
                        'text_id': text_id,
                    })
        self.data = data

    def _convert_gt_label(self):
        data = self.data
        gt_label = {item['id']: item['text'] for item in data}
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

        text_dict = {item['id']: item['text'] for item in self.data}
        text_sorted = [text_dict[k] for k in sorted(prediction.keys())]
        for i in range(len(decoded_preds)):
            if decoded_preds[i] == '':
                decoded_preds[i] = text_sorted[i]
        # Replace -100 in the labels as we can't decode them.
        # labels = np.where(labels != -100, labels, mid_processor.tokenizer.pad_token_id)
        # decoded_labels = mid_processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = labels

        # tot = 0
        # for i in range(len(decoded_preds)):
        #     res.append({'image_id': tot, 'caption': [decoded_preds[i]]})
        #     gts[tot] = [decoded_labels[i]]
        #     tot += 1
        # CiderD_scorer = CiderD(df='corpus', sigma=15)
        # cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
        # metrics.update(cider=cider_score)
        # print(metrics.value())

        rouge = lawrouge.Rouge()
        result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
        
        metric = {
            # 'cider_score': cider_score,
            'rouge_1': result['rouge-1']['f'],
            'rouge_2': result['rouge-2']['f'],
            'rouge_L': result['rouge-l']['f'],
        }

        return metric
    
    def out(self, prediction, path):
        pass

import numpy as np

import torch

from transformers import BertTokenizer
from transformers.file_utils import PaddingStrategy

class PretrainProcessorWOPad:
    def __init__(self, config):
        self.config = config
        self.tokenizer_path = self.config['path']
        self.max_input_length = self.config['max_input_length']
        self.max_target_length = self.config['max_target_length']
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        self.label_pad_id = self.config['label_pad_id']
        self.pad_id = self.config['pad_id']

    def __call__(self, item, sample):
        text = item['text']
        model_inputs = self.tokenizer(text, max_length=self.max_input_length, truncation=True, return_tensors="pt")
        sample['input_ids'] = model_inputs['input_ids']
        sample['attention_mask'] = model_inputs['attention_mask'].squeeze(0)
        
        sample['input_ids'] = sample['input_ids'].squeeze(0)
        if 'label' in item:
            target = item['label']
            target_inputs = self.tokenizer(target, max_length=self.max_target_length, truncation=True, return_tensors="pt")
            decoder_input_ids = self.shift_tokens_right(target_inputs['input_ids'], 0, 102)
            sample['decoder_input_ids'] = decoder_input_ids.squeeze(0)
            sample['labels'] = target_inputs['input_ids'].squeeze(0)

        sample['label_pad_id'] = self.label_pad_id
        sample['pad_id'] = self.pad_id



    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
    

class MLMProcessorWOPad:
    def __init__(self, config):
        self.config = config

        self.tokenizer_path = self.config['path']
        self.max_length = self.config['max_length']
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)

        self.special_num=len(self.tokenizer.all_special_tokens)
        self.vocab_size=self.tokenizer.vocab_size

        self.label_pad_id = self.config['label_pad_id']
        self.pad_id = self.config['pad_id']

    def truncate(self, a:list, maxLen):
        maxLen-=3#空留给cls sep sep
        assert maxLen>=0
        if len(a)>maxLen:#需要截断
            a=a[:maxLen]
        return a
    
    def random_mask(self, text_ids):
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        idx=0
        while idx<len(rands):
            if rands[idx]<0.15:#需要mask
                ngram=np.random.choice([1,2,3], p=[0.7,0.2,0.1])#若要mask，进行x_gram mask的概率
                if ngram==3 and len(rands)<7:#太大的gram不要应用于过短文本
                    ngram=2
                if ngram==2 and len(rands)<4:
                    ngram=1
                L=idx+1
                R=idx+ngram#最终需要mask的右边界（开）
                while L<R and L<len(rands):
                    rands[L]=np.random.random()*0.15#强制mask
                    L+=1
                idx=R
                if idx<len(rands):
                    rands[idx]=1#禁止mask片段的下一个token被mask，防止一大片连续mask
            idx+=1

        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input_ids.append(self.tokenizer.mask_token_id)
                output_ids.append(i)#mask预测自己
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)#自己预测自己
            elif r < 0.15:
                input_ids.append(np.random.randint(self.special_num,self.vocab_size))
                output_ids.append(i)#随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己
            else:
                input_ids.append(i)
                output_ids.append(-100)#保持原样不预测

        return input_ids, output_ids

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


    def __call__(self, item, sample):
        text = item['text']
        text = text.split()
        text= self.truncate(text, self.max_length)
        text_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_ids, out_ids = self.random_mask(text_ids)
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]#拼接
        labels = [-100] + out_ids + [-100]
        # assert len(input_ids)==len(labels)
        # sample['input_ids'] = input_ids
        # sample['decoder_input_ids'] = labels

        # model_inputs = self.tokenizer(text, max_length=self.max_input_length, truncation=True, return_tensors="pt")
        attention_mask = [1 for _ in range(len(input_ids))]
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
        sample['input_ids'] = input_ids
        sample['attention_mask'] = attention_mask
        
        # sample['input_ids'] = sample['input_ids'].squeeze(0)
        
        # target = item['label']
        # target_inputs = self.tokenizer(target, max_length=self.max_target_length, truncation=True, return_tensors="pt")
        target_inputs = torch.tensor(labels, dtype=torch.int64).unsqueeze(0)
        decoder_input_ids = self.shift_tokens_right(target_inputs, 0, 102)
        sample['decoder_input_ids'] = decoder_input_ids.squeeze(0)
        sample['labels'] = target_inputs.squeeze(0)

        sample['label_pad_id'] = self.label_pad_id
        sample['pad_id'] = self.pad_id

        # model_inputs = self.tokenizer(text, max_length=self.max_input_length, truncation=True, return_tensors="pt", padding=PaddingStrategy.MAX_LENGTH)
        # sample['input_ids'] = model_inputs['input_ids']
        # sample['attention_mask'] = model_inputs['attention_mask'].squeeze(0)
        
        # sample['input_ids'] = sample['input_ids'].squeeze(0)
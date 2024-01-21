from transformers import AutoTokenizer, BertTokenizer
from transformers.file_utils import PaddingStrategy
import torch

class MidProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer_path = self.config['path']
        self.max_input_length = self.config['max_input_length']
        self.max_target_length = self.config['max_target_length']
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)

    def __call__(self, item, sample):
        text = item['text']
        model_inputs = self.tokenizer(text, max_length=self.max_input_length, truncation=True, return_tensors="pt", padding=PaddingStrategy.MAX_LENGTH)
        sample['input_ids'] = model_inputs['input_ids']
        sample['attention_mask'] = model_inputs['attention_mask'].squeeze(0)
        
        sample['input_ids'] = sample['input_ids'].squeeze(0)
        if 'label' in item:
            target = item['label']
            target_inputs = self.tokenizer(target, max_length=self.max_target_length, truncation=True, return_tensors="pt", padding=PaddingStrategy.MAX_LENGTH)
            decoder_input_ids = self.shift_tokens_right(target_inputs['input_ids'], 0, 102)
            sample['decoder_input_ids'] = decoder_input_ids.squeeze(0)
            sample['labels'] = target_inputs['input_ids'].squeeze(0)



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


class MidProcessorAdd:
    def __init__(self, config):
        self.config = config
        self.tokenizer_path = self.config['path']
        self.max_input_length = self.config['max_input_length']
        self.max_target_length = self.config['max_target_length']
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)

    def __call__(self, item, sample):
        text = item['text']
        clinical = item['clinical']
        text += ' ' + clinical
        model_inputs = self.tokenizer(text, max_length=self.max_input_length, truncation=True, return_tensors="pt", padding=PaddingStrategy.MAX_LENGTH)
        sample['input_ids'] = model_inputs['input_ids']
        sample['attention_mask'] = model_inputs['attention_mask'].squeeze(0)
        
        sample['input_ids'] = sample['input_ids'].squeeze(0)
        if 'label' in item:
            target = item['label']
            target_inputs = self.tokenizer(target, max_length=self.max_target_length, truncation=True, return_tensors="pt", padding=PaddingStrategy.MAX_LENGTH)
            decoder_input_ids = self.shift_tokens_right(target_inputs['input_ids'], 0, 102)
            sample['decoder_input_ids'] = decoder_input_ids.squeeze(0)
            sample['labels'] = target_inputs['input_ids'].squeeze(0)



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


class MidProcessorWOPad:
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
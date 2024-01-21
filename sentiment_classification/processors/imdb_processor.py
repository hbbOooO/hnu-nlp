import torch

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.file_utils import PaddingStrategy

from typing import List

class Tokenizer:
    def __init__(self, config):
        self.config = config
        self.tokenizer_path = config['path']
        self.max_length = config['max_length']
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def _tokenize_more_sentence(self, sentences, sentence_num):
        raise NotImplementedError

    def _tokenize(self, sentences: List[str]):
        sentence_num = self.config['sentence_num']
        if sentence_num == 1:
            sentence_0 = sentences[0]
            results = self.tokenizer.encode_plus(
                sentence_0,
                add_special_tokens=True,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=TruncationStrategy.LONGEST_FIRST,
                max_length=self.max_length,
                return_tensors="pt",
                return_token_type_ids=True,
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_length=True
            )
        elif sentence_num == 2:
            sentence_0, sentence_1 = sentences[0], sentences[1]
            results = self.tokenizer.encode_plus(
                sentence_0,
                text_pair=sentence_1,
                add_special_tokens=True,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=TruncationStrategy.LONGEST_FIRST,
                max_length=self.max_length,
                return_tensors="pt",
                return_token_type_ids=True,
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_length=True
            )
        else:
            self._tokenize_more_sentence()
        return results

class SingleTokenizer(Tokenizer):
    def __init__(self, config):
        super().__init__(config)
        self.text_max_length = config['text_max_length']

    def __call__(self, item, sample):
        text_sentences = item['text_sentences']
        pad_text_sentences = ['pad' for _ in range(self.text_max_length)]
        pad_text_sentences[:min(len(text_sentences),self.text_max_length)] = text_sentences[:min(len(text_sentences),self.text_max_length)]
        sen_inds_list = []
        sen_mask_list = []
        for sen in pad_text_sentences:
            sen_results = self._tokenize([sen])
            sen_inds_list.append(sen_results['input_ids'].squeeze())
            sen_mask_list.append(sen_results['attention_mask'].squeeze())
        text_mask = torch.tensor([1 if i < len(text_sentences) else 0 for i in range(self.text_max_length)])
        sample['sen_inds_list'] = sen_inds_list
        sample['sen_mask_list'] = sen_mask_list
        sample['text_mask'] = text_mask

class IdProcessor:
    def __init__(self, config):
        self.config = config

    def __call__(self, item, sample):
        sample['id'] = item['id']
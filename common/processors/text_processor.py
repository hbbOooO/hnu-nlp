'''
文本预处理器
'''

import torch

from transformers import AutoTokenizer
# from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
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

    def __call__(self, item, sample):
        text = item['text']
        results = self._tokenize([text])
        sample['input_ids'] = results['input_ids'].squeeze(0)
        sample['attention_mask'] = results['attention_mask'].squeeze(0)

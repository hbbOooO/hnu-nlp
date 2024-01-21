import torch
from jieba import lcut

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.file_utils import PaddingStrategy

class CatProcessor:
    def __init__(self, config):
        self.config = config

    def __call__(self, item, sample):
            
        points = item['points']
        item['point'] = ''.join(points)


class TextProcessor:

    # INPUT_FORMAT = '主题：{}。'
    INPUT_FORMAT = '是的，综上所述，{}，谢谢。'
    INPUT_FORMAT_CLS = '主题：{}。角度：{}。'

    def __init__(self, config):
        self.config = config
        self.input_max_length = config['input_max_length']
        self.output_max_length = config['output_max_length']
        self.tokenizer_path = config['path']
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.add_angle = config['add_angle']
        

    def __call__(self, item, sample):
        use_argument_cls = item['use_argument_cls']
        if not use_argument_cls or not self.add_angle:
            input_text = self.INPUT_FORMAT.format(item['claim'])
        else:
            int_cls = item['int_cls']
            argument_cls = item['argument_cls']
            angles = '，'.join([argument_cls[i] for i in range(len(int_cls)) if int_cls[i] == '1'])
            input_text = self.INPUT_FORMAT_CLS.format(item['claim'], angles)
        input_results = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=TruncationStrategy.LONGEST_FIRST,
            max_length=self.input_max_length,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_length=True
        )
        sample['input_input_ids'] = input_results['input_ids'].squeeze(0)
        sample['input_attention_mask'] = input_results['attention_mask'].squeeze(0)

        if 'argument' in item:
            output_text = item['argument']
            output_results = self.tokenizer.encode_plus(
                output_text,
                add_special_tokens=True,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=TruncationStrategy.LONGEST_FIRST,
                max_length=self.output_max_length,
                return_tensors="pt",
                return_token_type_ids=True,
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_length=True
            )
            output_input_ids = output_results['input_ids']
            output_input_ids[torch.where(output_input_ids == self.tokenizer.pad_token_id)] = -100
            sample['output_input_ids'] = output_input_ids.squeeze(0)
            
            sample['output_attention_mask'] = output_results['attention_mask'].squeeze(0)


class TextProcessorPrompt:

    INPUT_FORMAT = '主题：{}。'
    
    INPUT_FORMAT_CLS = '主题：{}。角度：{}。'

    def __init__(self, config):
        self.config = config
        self.input_max_length = config['input_max_length']
        self.output_max_length = config['output_max_length']
        self.tokenizer_path = config['path']
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.add_angle = config['add_angle']
        self.val_angles = config['val_angles']
        

    def __call__(self, item, sample):
        use_argument_cls = item['use_argument_cls']
        if not use_argument_cls or not self.add_angle:
            input_text = self.INPUT_FORMAT.format(item['claim'])

            input_results = self.tokenizer.encode_plus(
                input_text,
                add_special_tokens=True,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=TruncationStrategy.LONGEST_FIRST,
                max_length=self.input_max_length,
                return_tensors="pt",
                return_token_type_ids=True,
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_length=True
            )
            sample['input_input_ids'] = input_results['input_ids'].squeeze(0)
            sample['input_attention_mask'] = input_results['attention_mask'].squeeze(0)

            if 'argument' in item:
                output_text = item['argument']
                output_results = self.tokenizer.encode_plus(
                    output_text,
                    add_special_tokens=True,
                    padding=PaddingStrategy.MAX_LENGTH,
                    truncation=TruncationStrategy.LONGEST_FIRST,
                    max_length=self.output_max_length,
                    return_tensors="pt",
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    return_special_tokens_mask=True,
                    return_length=True
                )
                output_input_ids = output_results['input_ids']
                output_input_ids[torch.where(output_input_ids == self.tokenizer.pad_token_id)] = -100
                sample['output_input_ids'] = output_input_ids.squeeze(0)
                
                sample['output_attention_mask'] = output_results['attention_mask'].squeeze(0)
        elif 'int_cls' in item:
            int_cls = item['int_cls']
            argument_cls = item['argument_cls']
            angles = '，'.join([argument_cls[i] for i in range(len(int_cls)) if int_cls[i] == '1'])
            input_text = self.INPUT_FORMAT_CLS.format(item['claim'], angles)

            input_results = self.tokenizer.encode_plus(
                input_text,
                add_special_tokens=True,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=TruncationStrategy.LONGEST_FIRST,
                max_length=self.input_max_length,
                return_tensors="pt",
                return_token_type_ids=True,
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_length=True
            )
            sample['input_input_ids'] = input_results['input_ids'].squeeze(0)
            sample['input_attention_mask'] = input_results['attention_mask'].squeeze(0)

            if 'argument' in item:
                output_text = item['argument']
                output_results = self.tokenizer.encode_plus(
                    output_text,
                    add_special_tokens=True,
                    padding=PaddingStrategy.MAX_LENGTH,
                    truncation=TruncationStrategy.LONGEST_FIRST,
                    max_length=self.output_max_length,
                    return_tensors="pt",
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    return_special_tokens_mask=True,
                    return_length=True
                )
                output_input_ids = output_results['input_ids']
                output_input_ids[torch.where(output_input_ids == self.tokenizer.pad_token_id)] = -100
                sample['output_input_ids'] = output_input_ids.squeeze(0)
                
                sample['output_attention_mask'] = output_results['attention_mask'].squeeze(0)
        else:
            input_texts = [self.INPUT_FORMAT_CLS.format(item['claim'], val_angle) for val_angle in self.val_angles]
            input_input_ids_list = []
            input_attention_mask_list = []
            for input_text in input_texts:
                input_results = self.tokenizer.encode_plus(
                    input_text,
                    add_special_tokens=True,
                    padding=PaddingStrategy.MAX_LENGTH,
                    truncation=TruncationStrategy.LONGEST_FIRST,
                    max_length=self.input_max_length,
                    return_tensors="pt",
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    return_special_tokens_mask=True,
                    return_length=True
                )
                input_input_ids_list.append(input_results['input_ids'].squeeze(0))
                input_attention_mask_list.append(input_results['attention_mask'].squeeze(0))
                sample['input_input_ids_list'] = input_input_ids_list
                sample['input_attention_mask_list'] = input_attention_mask_list
        

class W2vProcessor:
    def __init__(self, config):
        self.condif = config
        import fasttext
        fasttext_path = config['fasttext_path'] # '/root/autodl-tmp/word2vec_model/cc.zh.300.bin.gz'
        self.fasttext_model = fasttext.load_model(fasttext_path)
        vocab_path = config['vocab_path']
        with open(vocab_path) as f:
            vocabs = f.readlines()
        vocabs = [vocab[:-1] for vocab in vocabs]
        self.vocabs = vocabs
        self.vocab_vectors = self.fasttext_model(vocabs)
        self.vocab_len = len(vocabs)
        self.max_claim_len = config['max_claim_len']
        self.max_label_len = config['max_label_len']

        self.pad_id = 0
        self.unk_id = 1
        self.sat_id = 2
        self.end_id = 3
        self.pad_token = vocabs[self.pad_id]
        self.unk_token = vocabs[self.unk_id]
        self.sat_token = vocabs[self.sat_id]
        self.end_token = vocabs[self.end_id]
        self.special_tokens = [self.pad_token, self.unk_token, self.sat_token, self.end_token]
    
    def __call__(self, item, sample):
        claim = item['claim']
        claim_tokens = lcut(claim)
        claim_tokens_pad = [claim_tokens[i] if i < len(claim_tokens) else self.pad_token for i in range(self.max_claim_len)]
        claim_mask = [1 if i < len(claim_tokens) else 0 for i in range(self.max_claim_len)]
        claim_token_vectors = self.fasttext_model(claim_tokens_pad)
        claim_mask = torch.tensor(claim_mask)
        claim_token_vectors = torch.tensor(claim_token_vectors)

        sample['claim_mask'] = claim_mask
        sample['claim_token_vectors'] = claim_token_vectors

        claim_vocab = {claim_tokens[i]: i+self.vocab_len for i in range(len(claim_tokens))}

        if 'label' in item:
            label = item['label']
            label_tokens = lcut(label)
            label_tokens_spec = [self.sat_token] + label_tokens + [self.end_token]
            label_tokens_pad = [label_tokens_spec[i] if i < len(label_tokens_spec) else self.pad_token for i in range(self.max_label_len)]
            label_mask = [1 if i < len(label_tokens_spec) else 0 for i in range(self.max_label_len)]
            label_token_vectors = self.fasttext_model(label_tokens_pad)
            label_mask = torch.tensor(label_mask)
            label_token_vectors = torch.tensor(label_token_vectors)

            sample['label_mask'] = label_mask
            sample['label_token_vectors'] = label_token_vectors

            label_id = [self.vocabs.index(token) if token in self.vocab else claim_vocab[token] for token in label_tokens]
            label_id = torch.tensor(label_id)
            sample['label'] = label_id
        else:
            label_tokens = [self.pad_token for _ in range(self.max_label_len)]
            label_tokens[0] = self.sat_token
            label_token_vectors = self.fasttext_model(label_tokens_pad)
            label_token_vectors = torch.tensor(label_token_vectors)

            sample['label_token_vectors'] = label_token_vectors
        

    def decode(self, pred_ids, claim, skip_special_tokens=True):
        claim_tokens = lcut(claim)
        # claim_vocab = {claim_tokens[i]: i+self.vocab_len for i in range(len(claim_tokens))}
        ans = [self.vocabs[id] if id < self.vocab_len else claim_tokens[id-self.vocab_len] for id in pred_ids]
        if skip_special_tokens:
            ans = [word for word in ans if word not in self.special_tokens]
        return ''.join(ans)

    # def iteration_decode(self, pred_batch, )

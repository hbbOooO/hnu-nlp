import torch

from transformers import AutoTokenizer, BertTokenizer

class CscProcessorWOPad:
    def __init__(self, config):
        self.config = config
        self.tokenizer_path = self.config['path']
        # self.max_input_length = self.config['max_input_length']
        # self.max_target_length = self.config['max_target_length']
        # self.max_length = self.config['max_length']
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        self.pad_id = config['pad_id']
        self.label_pad_id = config['label_pad_id']

    def __call__(self, item, sample):
        ori_text = item['original_text']
        cor_text = item.get('correct_text', None)
        wrong_ids = item.get('wrong_ids', None)

        if cor_text is not None:

            encoded_text = self.tokenizer.tokenize(ori_text)
            det_label = torch.zeros(len(encoded_text) + 2).long()
            for idx in wrong_ids:
                margins = []
                for word in encoded_text[:idx]:
                    # if word == 'UNK':
                    if word == self.tokenizer.unk_token:
                        break
                    if word.startswith('##'):
                        margins.append(len(word) - 3)
                    else:
                        margins.append(len(word) - 1)
                margin = sum(margins)
                move = 0
                while (abs(move) < margin) or (idx + move >= len(encoded_text)) \
                        or encoded_text[idx + move].startswith('##'):
                    move -= 1
                det_label[idx + move + 1] = 1
            
            text_label = self.tokenizer(cor_text, return_tensors="pt")['input_ids']
        
            sample['text_label'] = text_label.squeeze(0)
            sample['det_label'] = det_label

        tokenized_input = self.tokenizer(ori_text, return_tensors="pt")
        sample['input_ids'] = tokenized_input['input_ids'].squeeze(0)
        sample['attention_mask'] = tokenized_input['attention_mask'].squeeze(0)
        sample['token_type_ids'] = tokenized_input['token_type_ids'].squeeze(0)

        sample['pad_id'] = self.pad_id
        sample['label_pad_id'] = self.label_pad_id
        

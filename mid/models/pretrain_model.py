import torch
import os

from common.models.base_model import BaseModel
from common.modules.auto_net import AutoNetForSeq2SeqLM

class PretrainModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.auto_net = AutoNetForSeq2SeqLM(config['auto_net'])
        self.auto_net.auto_net.resize_token_embeddings(self.config['vocab_size'])

        self.max_length = self.config['max_length'] # 80
        self.num_beams = self.config['num_beams'] # 5
        self.synced_gpus = self.config['synced_gpus'] # False

    def forward_train(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        decoder_input_ids = batch['decoder_input_ids']
        labels = batch['labels']

        outputs = self.auto_net(**{
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids
        })

        # 暂时没有实现 损失函数 平滑曲线
        # loss = self.label_smoother(outputs, labels)

        loss_input = {
            'pred': outputs,
            'label': labels
        }

        # pred = logistic.argmax(dim=-1)
        # pred_w_index = torch.cat([batch['id'].unsqueeze(-1), pred.unsqueeze(-1)], dim=-1)

        pred_w_index = None

        return loss_input, pred_w_index

    def forward_test(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # decoder_input_ids = batch['decoder_input_ids']

        generated_tokens = self.auto_net.auto_net.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            num_beams=self.num_beams,
            synced_gpus=self.synced_gpus
        )

        loss_input = {}
        pred = generated_tokens
        # pred_w_index = torch.cat([batch['id'].unsqueeze(-1), pred], dim=-1)
        pred_w_index = {
            'id': batch['id'],
            'pred': pred
        }

        return loss_input, pred_w_index
    
    def save_module(self, module_save_path):
        if not os.path.exists(module_save_path): os.makedirs(module_save_path)
        self.auto_net.auto_net.save_pretrained(module_save_path)



class MLMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.auto_net = AutoNetForSeq2SeqLM(config['auto_net'])
        self.auto_net.auto_net.resize_token_embeddings(self.config['vocab_size'])

        self.max_length = self.config['max_length'] # 80
        self.num_beams = self.config['num_beams'] # 5
        self.synced_gpus = self.config['synced_gpus'] # False

    def forward_train(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # decoder_input_ids = batch['decoder_input_ids']
        labels = batch['labels']

        outputs = self.auto_net(**{
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            # 'decoder_input_ids': decoder_input_ids
        })

        loss_input = {
            'loss': outputs['loss']
        }

        pred_w_index = None

        return loss_input, pred_w_index

    def forward_test(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # decoder_input_ids = batch['decoder_input_ids']

        generated_tokens = self.auto_net.auto_net.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            num_beams=self.num_beams,
            synced_gpus=self.synced_gpus
        )

        loss_input = {}
        pred = generated_tokens
        pred_w_index = {
            'id': batch['id'],
            'pred': pred
        }
        # pred_w_index = torch.cat([batch['id'].unsqueeze(-1), pred], dim=-1)

        return loss_input, pred_w_index
    
    def save_module(self, module_save_path):
        if not os.path.exists(module_save_path): os.makedirs(module_save_path)
        self.auto_net.auto_net.save_pretrained(module_save_path)
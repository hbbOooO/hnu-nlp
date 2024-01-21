import torch
from torch import nn

import os

from common.models.base_model import BaseModel
from common.modules.auto_net import AutoNetForMaskedLM

class CscModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.auto_net = AutoNetForMaskedLM(config['auto_net'])

        self.detection = nn.Linear(self.auto_net.auto_net.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_train(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        text_label = batch['text_label']
        det_label = batch['det_label']

        model_outputs = self.auto_net(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=text_label,
            return_dict=True,
            output_hidden_states=True
        )

        prob = self.detection(model_outputs.hidden_states[-1])

        active_loss = attention_mask.view(-1, prob.shape[1]) == 1
        active_probs = prob.view(-1, prob.shape[1])[active_loss]
        active_labels = det_label[active_loss].float()

        word_loss = model_outputs.loss

        loss_input = {
            'pred':{ 
                'word_loss': word_loss,
                'det_prob': active_probs
            }, 
            'label':{
                'det_label': active_labels
            }
        }
        
        det_prob = self.sigmoid(prob).squeeze(-1)
        word_prob = model_outputs.logits

        det_pred = (det_prob > 0.5).long()
        word_pred = torch.argmax(word_prob, dim=-1)

        pred_w_index = {
            'id': batch['id'],
            'det_pred': det_pred,
            'word_pred': word_pred
        }
        if text_label is not None: pred_w_index['text_label'] = text_label
        if det_label is not None: pred_w_index['det_label'] = det_label

        return loss_input, pred_w_index

    def forward_test(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        text_label = batch.get('text_label', None)
        det_label = batch.get('det_label', None)

        model_outputs = self.auto_net(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # labels=text_label,
            return_dict=True,
            output_hidden_states=True
        )
        prob = self.detection(model_outputs.hidden_states[-1])

        det_prob = self.sigmoid(prob).squeeze(-1)
        word_prob = model_outputs.logits

        det_pred = (det_prob > 0.5).long()
        word_pred = torch.argmax(word_prob, dim=-1)

        loss_input = {}
        # pred = generated_tokens
        pred_w_index = {
            'id': batch['id'],
            'det_pred': det_pred,
            'word_pred': word_pred
        }
        if text_label is not None: pred_w_index['text_label'] = text_label
        if det_label is not None: pred_w_index['det_label'] = det_label
        # pred_w_index = torch.cat([batch['id'].unsqueeze(-1), pred], dim=-1)

        return loss_input, pred_w_index

    def save_module(self, module_save_path):
        if not os.path.exists(module_save_path): os.makedirs(module_save_path)
        torch.save(self.state_dict(), module_save_path)
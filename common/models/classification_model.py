import torch
from torch import nn
from torch.nn import functional as F

from common.modules.auto_net import AutoNetForSequenceClassification
from common.models.base_model import BaseModel

class ClassificationModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.auto_net = AutoNetForSequenceClassification(config['auto_net'])
        self.cal_loss_by_autonet = self.config.get('cal_loss_by_autonet', False)
        
    def forward_both(self, batch):
        res = self.auto_net(**batch)
        batch['res'] = res

    
    def forward_train(self, batch):
        if self.cal_loss_by_autonet:
            loss_input = {
                'loss': batch['res']['loss']
            }
        else:
            loss_input = {
                'pred': batch['res']['logits'],
                'label': batch['labels']
            }
        pred = batch['res']['logits'].argmax(dim=-1)
        pred_w_index = {
            'id': batch['id'],
            'pred': pred
        }

        return loss_input, pred_w_index

    def forward_test(self, batch):
        logistic = batch['res']['logits']
        loss_input = {}
        pred = logistic.argmax(dim=-1)
        pred_w_index = {
            'id': batch['id'],
            'pred': pred
        }

        return loss_input, pred_w_index



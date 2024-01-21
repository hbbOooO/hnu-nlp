import torch
from torch import nn
from torch.nn import functional as F

from common.modules.auto_net import AutoNet
from common.models.base_model import BaseModel

class IMDBModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.auto_net = AutoNet(config['auto_net'])

        self.drop_out = nn.Dropout(config['drop_out'])
        self.classifier = nn.Linear(self.auto_net.auto_net_config.hidden_size, config['cls_num'])
        
    def forward_train(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        feat = self.auto_net(**{'input_ids':input_ids,'attention_mask':attention_mask})
        feat = feat['pooler_output']
        feat = self.drop_out(feat)
        logistic = self.classifier(feat)
        logistic = F.softmax(logistic, dim=-1)
        loss_input = {
            'pred': logistic,
            'label': batch['label']
        }
        pred = logistic.argmax(dim=-1)
        pred_w_index = torch.cat([batch['id'].unsqueeze(-1), pred.unsqueeze(-1)], dim=-1)

        return loss_input, pred_w_index

    def forward_test(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        feat = self.auto_net(**{'input_ids':input_ids,'attention_mask':attention_mask})
        feat = feat['pooler_output']
        feat = self.drop_out(feat)
        logistic = self.classifier(feat)
        logistic = F.softmax(logistic, dim=-1)
        loss_input = {}
        pred = logistic.argmax(dim=-1)
        pred_w_index = torch.cat([batch['id'].unsqueeze(-1), pred.unsqueeze(-1)], dim=-1)

        return loss_input, pred_w_index


    


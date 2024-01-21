import torch

from common.models.base_model import BaseModel
from common.modules.auto_net import AutoNetForSeq2SeqLM

class GenerationModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.auto_net = AutoNetForSeq2SeqLM(config['auto_net'])
        self.pred_config = self.config['pred_config']
        self.cal_loss_by_autonet = self.config.get('cal_loss_by_autonet', False)

    def forward_both(self, batch):
        pass

    def forward_train(self, batch):
        res = self.auto_net(**batch)
        batch['res'] = res
        if self.cal_loss_by_autonet:
            loss_input = {
                'loss': batch['res']['loss']
            }
        else:
            loss_input = {
                'pred': batch['res']['logits'],
                'label': batch['labels']
            }

        pred_w_index = None

        return loss_input, pred_w_index

    def forward_test(self, batch):
        # input_ids = batch['input_ids']
        # attention_mask = batch['attention_mask']
        # decoder_input_ids = batch['decoder_input_ids']

        batch.update(self.pred_config)

        generated_tokens = self.auto_net.generate(
            **batch
        )

        loss_input = {}
        pred = generated_tokens
        pred_w_index = {
            'id': batch['id'],
            'pred': pred
        }

        return loss_input, pred_w_index



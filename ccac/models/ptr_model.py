import torch
import functools

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.bert.configuration_bert import BertConfig

from ccac.utils.point_net import PointerGeneratorNetworks
from common.models.base_model import BaseModel


class PtrModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self_att_config = BertConfig(config['self_att'])
        self.self_att = BertSelfAttention(self_att_config)

        self.seq_linear = nn.Linear(config['emb_dim'], config['hidden_dim'])
        self.dec_linear = nn.Linear(config['emb_dim'], config['hidden_dim'])

        import fasttext
        fasttext_path = config['fasttext_path'] # '/root/autodl-tmp/word2vec_model/cc.zh.300.bin.gz'
        self.fasttext_model = fasttext.load_model(fasttext_path)


    def forward_both(self, batch):
        claim_token_vectors = batch['claim_token_vectors']
        claim_feat = self.seq_linear(claim_token_vectors)
        # claim_mask = batch['claim_mask']
        batch['claim_feat'] = claim_feat
        # batch['claim_mask_expand'] = claim_mask_expand

    def forward_train(self, batch):
        label_token_vectors = batch['label_token_vectors']
        label_feat = self.dec_linear(label_token_vectors)
        label_mask = batch['label_mask']
        self_att_mask = _get_causal_mask(label_feat.size(1), label_feat.device)
        self_att_output = self.self_att(label_feat, attention_mask=self_att_mask)
        logistics = torch.bmm(self_att_output, batch['claim_feat'])
        score_mask = torch.bmm(label_mask.unsqueeze(-1), batch['claim_mask'].unsqueeze(1))
        # logistics += (1 - logistics_mask) * -10000

        scores = F.softmax(logistics, dim=-1)
        masked_scores = scores * score_mask
        # logistics += (1 - label_mask) * -10000

        pred = scores.argmax(dim=-1)

        pred_w_index = {
            'id': batch['id'],
            'pred': pred
        }

        loss_input = {
            'pred': masked_scores,
            'label': batch['label']
        }

        return loss_input, pred_w_index
        

    def forward_test(self, batch):
        label_token_vectors = batch['label_token_vectors']
        label_feat = label_token_vectors
        for t in range(label_token_vectors.size(1)):
            label_feat = self.dec_linear(label_feat)
            self_att_mask = _get_causal_mask(label_feat.size(1), label_feat.device)
            self_att_output = self.self_att(label_feat, attention_mask=self_att_mask)
            logistics = torch.bmm(self_att_output, batch['claim_feat'])
            scores = F.softmax(logistics, dim=-1)
            pred = scores.argmax(dim=-1)

            fwd_results['prev_inds'][:, 1:] = pred[:, :-1]

@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 1.
    return mask



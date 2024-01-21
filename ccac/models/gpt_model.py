import torch

from common.models.base_model import BaseModel
from common.modules.auto_net import AutoNetForLMHead
from transformers import AutoConfig, GPT2LMHeadModel

class GPTModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # self.auto_net = AutoNetForLMHead(config['auto_net'])
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net']['auto_net_path'], **config['auto_net']['auto_net_config'])
        self.auto_net = GPT2LMHeadModel.from_pretrained(config['auto_net']['auto_net_path'], config=self.auto_net_config)
        self.max_length = config['max_length']
        self.num_beams = config['num_beams']
        # self.synced_gpus = config['synced_gpus']
        # self.repetition_penalty = config['repetition_penalty']
        self.eos_token_id = config['eos_token_id']
        self.add_prompt = config['add_prompt']


    def forward_both(self, batch):
        return batch

    def forward_train(self, batch):
        input_ids = batch['input_input_ids']
        attention_mask = batch['input_attention_mask']
        labels = batch['output_input_ids']
        # decoder_attention_mask = batch['output_attention_mask']

        outputs = self.auto_net(**{
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            # 'decoder_attention_mask': decoder_attention_mask
        })

        pred_w_index = None

        loss = outputs['loss']
        loss_input = {
            'loss': loss
        }

        return loss_input, pred_w_index

    def forward_test(self, batch):
        if not self.add_prompt:
            input_ids = batch['input_input_ids']
            attention_mask = batch['input_attention_mask']
            batch_size = input_ids.size(0)

            generated_tokens_list = []
            for _ in range(5):
                generated_tokens = self.auto_net.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    # synced_gpus=self.synced_gpus,
                    # repetition_penalty=self.repetition_penalty,
                    eos_token_id=self.eos_token_id, 
                    pad_token_id=self.eos_token_id
                )
                generated_tokens_list.append(generated_tokens)
            generated_tokens_0 = generated_tokens_list[0]
            generated_tokens_1 = generated_tokens_list[1]
            generated_tokens_2 = generated_tokens_list[2]
            generated_tokens_3 = generated_tokens_list[3]
            generated_tokens_4 = generated_tokens_list[4]

            # generated_tokens_list = self.auto_net.auto_net.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     max_length=self.max_length,
            #     num_beams=self.num_beams,
            #     # synced_gpus=self.synced_gpus,
            #     repetition_penalty=self.repetition_penalty,
            #     num_return_sequences = 5
            # )
            # generated_tokens_0 = generated_tokens_list[:batch_size]
            # generated_tokens_1 = generated_tokens_list[batch_size:batch_size*2]
            # generated_tokens_2 = generated_tokens_list[batch_size*2:batch_size*3]
            # generated_tokens_3 = generated_tokens_list[batch_size*3:batch_size*4]
            # generated_tokens_4 = generated_tokens_list[batch_size*4:]
        else:
            input_ids_list = batch['input_input_ids_list']
            attention_mask_list = batch['input_attention_mask_list']
            generated_tokens_list = []
            for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                generated_tokens = self.auto_net.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    # repetition_penalty=self.repetition_penalty,
                    eos_token_id=self.eos_token_id, 
                    pad_token_id=self.eos_token_id
                )
                generated_tokens_list.append(generated_tokens)
            generated_tokens_0 = generated_tokens_list[0]
            generated_tokens_1 = generated_tokens_list[1]
            generated_tokens_2 = generated_tokens_list[2]
            generated_tokens_3 = generated_tokens_list[3]
            generated_tokens_4 = generated_tokens_list[4]


        loss_input = {}
        pred_w_index = {
            'id': batch['id'],
            'pred_0': generated_tokens_0,
            'pred_1': generated_tokens_1,
            'pred_2': generated_tokens_2,
            'pred_3': generated_tokens_3,
            'pred_4': generated_tokens_4,
            'input': input_ids
        }

        return loss_input, pred_w_index
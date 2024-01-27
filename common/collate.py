"""
date: 2024/1/27
author: Mingjie Han
Describe: 重写dataloader类的collate方法
"""
import torch


def collate_item(batch, id, name):
    max_length = max([item[name].size(0) for item in batch])
    item_list = []
    for item in batch:
        pad_length = max_length - item[name].size(0)
        pad_item = torch.cat([item[name], torch.tensor([id for _ in range(pad_length)])])
        item_list.append(pad_item.unsqueeze(0))
    return torch.cat(item_list, dim=0).to(dtype=torch.int64)


def mid_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    label_pad_id = batch[0]['label_pad_id']
    pad_id = batch[0]['pad_id']

    collated_batch = {}

    if 'labels' in batch[0]:
        collated_batch['labels'] = collate_item(batch, label_pad_id, 'labels')

        collated_batch['decoder_input_ids'] = collate_item(batch, pad_id, 'decoder_input_ids')

    collated_batch['input_ids'] = collate_item(batch, pad_id, 'input_ids')

    collated_batch['attention_mask'] = collate_item(batch, pad_id, 'attention_mask')

    collated_batch['id'] = torch.tensor([item['id'] for item in batch])

    return collated_batch


def csc_collate(batch):
    label_pad_id = batch[0]['label_pad_id']
    pad_id = batch[0]['pad_id']

    collated_batch = {}

    if 'text_label' in batch[0]:
        collated_batch['text_label'] = collate_item(batch, label_pad_id, 'text_label')

        collated_batch['det_label'] = collate_item(batch, pad_id, 'det_label')

    collated_batch['input_ids'] = collate_item(batch, pad_id, 'input_ids')
    collated_batch['attention_mask'] = collate_item(batch, pad_id, 'attention_mask')
    collated_batch['token_type_ids'] = collate_item(batch, pad_id, 'token_type_ids')

    collated_batch['id'] = torch.tensor([item['id'] for item in batch])

    return collated_batch
    

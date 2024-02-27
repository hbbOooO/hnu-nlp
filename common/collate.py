"""
date: 2024/1/27
author: Mingjie Han
Describe: 重写dataloader类的collate方法。提供不定长的补空方式。
"""
import torch

from typing import Dict, Any



def collate_item(batch: Dict[str, Any], pad_id: int, name: str) -> torch.Tensor:
    """根据这个Batch中的名称为name的数据的最大长度，补空这一个Batch的数据

    Args:
        batch (Dict[str, Any]): 一个Batch的所有数据
        pad_id (int): 补空的数字编号
        name (str): 要Batch化的数据名称。因为有些数据不需要传入model中。

    Return:
        batched data (torch.Tensor): 已经Batch化的数据
    
    """
    max_length = max([item[name].size(0) for item in batch])
    item_list = []
    for item in batch:
        pad_length = max_length - item[name].size(0)
        pad_item = torch.cat([item[name], torch.tensor([pad_id for _ in range(pad_length)])])
        item_list.append(pad_item.unsqueeze(0))
    return torch.cat(item_list, dim=0).to(dtype=torch.int64)


def batch_padding_collate(batch: Dict[str, Any]) -> Dict[str, Any]:
    """不定长的补空方式。Torch默认的补空方式为：整个Dataset中的所有数据都一样的长度，例如所有文本都\n
    补齐到512维度。这里采用了不定长的补空方式，一个Batch中的数据保持一样的长度，不同Batch之间的长度\n
    不一定一样。这样做的意义在于能节省显存。

    Args:
        batch (Dict[str, Any]): 一个Batch的所有数据

    Return:
        collated_batch (Dict[str, Any]): Batch化的数据
    
    """
    assert 'label_pad_id' in batch[0] and 'pad_id' in batch[0], "There must exist keys of 'label_pad_id' and 'pad_id'."

    label_pad_id = batch[0]['label_pad_id']
    pad_id = batch[0]['pad_id']

    collated_batch = {}

    label_pad_keys = batch[0]['label_pad_keys']       # 使用 label_pad_id 补空的数据的名称
    default_pad_keys = batch[0]['default_pad_keys']   # 使用 pad_id 补空的数据的名称

    for key in label_pad_keys:
        if key not in batch[0]: continue
        collated_batch[key] = collate_item(batch, label_pad_id, key)
    for key in default_pad_keys:
        if key not in batch[0]: continue
        collated_batch[key] = collate_item(batch, pad_id, key)

    collated_batch['id'] = torch.tensor([item['id'] for item in batch])

    return collated_batch


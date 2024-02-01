"""
date: 2024/1/26
author: Fuzhong Suo
Describe: 实现auto_net模块
"""
from typing import Dict, Any

import torch
from torch import nn, Tensor

from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM,
                          AutoModelForMaskedLM, AutoModelForSequenceClassification,
                          AutoModelWithLMHead)


class AutoNet(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化工作，根据配置文件为模块进行装配
        :param config: (Dict[str, Any]) 模块配置字典
        """
        super().__init__()
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModel.from_pretrained(config['auto_net_path'], config=self.auto_net_config)

    def forward(self, *args, **kwargs) -> Tensor:
        """
        前向传播
        *args,**kwargs:接受任意数量的位置参数和关键字参数。
        :param args: 接受任意数量的位置参数
        :param kwargs: 接受任意数量的关键字参数
        :return:Tensor
        """
        res = self.auto_net(**kwargs)
        return res


class AutoNetForSeq2SeqLM(nn.Module):
    def __init__(self, config: (Dict[str, Any])) -> None:
        """
        初始化工作，根据配置文件为模块进行装配
        :param config:(Dict[str, Any]) 模块配置字典
        """
        super().__init__()
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelForSeq2SeqLM.from_pretrained(config['auto_net_path'], config=self.auto_net_config)

    def forward(self, *args, **kwargs) -> Tensor:
        """
        前向传播
        *args,**kwargs:接受任意数量的位置参数和关键字参数。
        :param args: 接受任意数量的位置参数
        :param kwargs: 接受任意数量的关键字参数
        :return:Tensor
        """
        res = self.auto_net(**kwargs)
        return res

    def generate(self, *args, **kwargs) -> Tensor:
        """
        生成内容
        :param args: 接受任意数量的位置参数
        :param kwargs: 接受任意数量的关键字参数
        :return: Tensor
        """
        res = self.auto_net.generate(*kwargs)
        return res


class AutoNetForMaskedLM(nn.Module):
    def __init__(self, config: (Dict[str, Any])) -> None:
        """
        初始化工作，根据配置文件为模块进行装配
        :param config:(Dict[str, Any]) 模块配置字典
        """
        super().__init__()
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelForMaskedLM.from_pretrained(config['auto_net_path'], config=self.auto_net_config)

    def forward(self, *args, **kwargs) -> Tensor:
        """
        前向传播
        :param args: 接受任意数量的位置参数
        :param kwargs: 接受任意数量的关键字参数
        :return: Tensor
        """
        res = self.auto_net(**kwargs)
        return res


class AutoNetForSequenceClassification(nn.Module):
    def __init__(self, config: (Dict[str, Any])) -> None:
        """
        始化工作，根据配置文件为模块进行装配
        :param config: config (Dict[str, Any]) 模块配置字典
        """
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelForSequenceClassification.from_pretrained(config['auto_net_path'],
                                                                           config=self.auto_net_config)

    def forward(self, *args, **kwargs) -> Tensor:
        """
        前向传播
        :param args: 接受任意数量的位置参数
        :param kwargs: 接受任意数量的关键字参数
        :return: Tensor
        """
        res = self.auto_net(**kwargs)
        return res


class AutoNetForLMHead(nn.Module):
    def __init__(self, config: (Dict[str, Any])) -> None:
        """
        始化工作，根据配置文件为模块进行装配
        :param config: config (Dict[str, Any]) 模块配置字典
        """
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelWithLMHead.from_pretrained(config['auto_net_path'], config=self.auto_net_config)

    def forward(self, *args, **kwargs) -> Tensor:
        """
        前向传播
        :param args: 接受任意数量的位置参数
        :param kwargs: 接受任意数量的关键字参数
        :return: Tensor
        """
        res = self.auto_net(**kwargs)
        return res
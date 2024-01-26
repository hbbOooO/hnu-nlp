"""
date: 2024/1/26
author: Fuzhong Suo
Describe: 实现auto_net模块
"""


import torch
from torch import nn

from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM,
                          AutoModelForMaskedLM, AutoModelForSequenceClassification,
                          AutoModelWithLMHead)

class AutoNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModel.from_pretrained(config['auto_net_path'], config=self.auto_net_config)

    """初始化工作，根据配置文件为模块进行装配

    Args:
        config (Dict[str, Any]): 模块配置字典
    """
    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    '''前向传播
    Args:
        *args,**kwargs:接受任意数量的位置参数和关键字参数。
    '''
class AutoNetForSeq2SeqLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelForSeq2SeqLM.from_pretrained(config['auto_net_path'], config=self.auto_net_config)
    """初始化工作，根据配置文件为模块进行装配

    Args:
        config (Dict[str, Any]): 模块配置字典
    """
    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    '''前向传播
    Args:
        *args,**kwargs:接受任意数量的位置参数和关键字参数。
    '''
    def generate(self, *args, **kwargs):
        res = self.auto_net.generate(*kwargs)
        return res
    '''生成内容
    Args:
        *args,**kwargs:接受任意数量的位置参数和关键字参数。
    '''

class AutoNetForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelForMaskedLM.from_pretrained(config['auto_net_path'], config=self.auto_net_config)
    """初始化工作，根据配置文件为模块进行装配

    Args:
        config (Dict[str, Any]): 模块配置字典
    """
    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    '''前向传播
    Args:
        *args,**kwargs:接受任意数量的位置参数和关键字参数。
    '''

class AutoNetForSequenceClassification(nn.Module):
    def __init__(self, config):
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelForSequenceClassification.from_pretrained(config['auto_net_path'], config=self.auto_net_config)
    """初始化工作，根据配置文件为模块进行装配

    Args:
        config (Dict[str, Any]): 模块配置字典
    """
    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    '''前向传播
    Args:
        *args,**kwargs:接受任意数量的位置参数和关键字参数。
    '''
class AutoNetForLMHead(nn.Module):
    def __init__(self, config):
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelWithLMHead.from_pretrained(config['auto_net_path'], config=self.auto_net_config)
    """初始化工作，根据配置文件为模块进行装配

    Args:
        config (Dict[str, Any]): 模块配置字典
    """
    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    '''生成内容
    Args:
        *args,**kwargs:接受任意数量的位置参数和关键字参数。
    '''

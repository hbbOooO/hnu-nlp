'''
date: 2024/1/26
author: Fuzhong Suo
Describe: 使用HuggingFace的模型的模块
'''

import torch
from torch import nn

from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM, 
                          AutoModelForMaskedLM, AutoModelForSequenceClassification,
                          AutoModelWithLMHead)

'''
Describe : 基础AutoNet模型
'''
class AutoNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModel.from_pretrained(config['auto_net_path'], config=self.auto_net_config)
    '''初始化工作，通过配置文件对模型进行初始化
    Args:
        config (Dict[str,Any]):模型配置字典
    '''
    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    '''前向传播
    Args：
        可以接受任意数量的位置参数和关键字参数
    '''

'''
Describe : Seq2Seq AutoNet模型
'''
class AutoNetForSeq2SeqLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelForSeq2SeqLM.from_pretrained(config['auto_net_path'], config=self.auto_net_config)
    '''初始化工作，通过配置文件对模型进行初始化
    Args:
        config(Dict[str,Any]):模型配置字典
    '''

    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    '''前向传播
    Args：
        可以接受任意数量的位置参数和关键字参数
    '''

    def generate(self, *args, **kwargs):
        res = self.auto_net.generate(*kwargs)
        return res
    '''生成文本
    Args：
        可以接受任意数量的位置参数和关键字参数
    '''


'''
Describe : AutoNetMaskedLM模型
'''
class AutoNetForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelForMaskedLM.from_pretrained(config['auto_net_path'], config=self.auto_net_config)
    '''初始化工作，通过配置文件对模型进行初始化
    Args:
        config(Dict[str,Any]):模型配置字典
    '''

    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    '''前向传播
    Args：
        可以接受任意数量的位置参数和关键字参数
    '''

class AutoNetForSequenceClassification(nn.Module):
    def __init__(self, config):
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelForSequenceClassification.from_pretrained(config['auto_net_path'], config=self.auto_net_config)
    '''初始化工作，通过配置文件对模型进行初始化
    Args:
        config(Dict[str,Any]):模型配置字典
    '''

    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    '''前向传播
    Args：
        可以接受任意数量的位置参数和关键字参数
    '''

class AutoNetForLMHead(nn.Module):
    def __init__(self, config):
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelWithLMHead.from_pretrained(config['auto_net_path'], config=self.auto_net_config)
    '''初始化工作，通过配置文件对模型进行初始化
    Args:
        config(Dict[str,Any]):模型配置字典
    '''

    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    '''前向传播
        Args：
            可以接受任意数量的位置参数和关键字参数
        '''
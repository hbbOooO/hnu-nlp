'''
使用HuggingFace的模型的模块
'''

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
    
    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res

class AutoNetForSeq2SeqLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelForSeq2SeqLM.from_pretrained(config['auto_net_path'], config=self.auto_net_config)
    
    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    
    def generate(self, *args, **kwargs):
        res = self.auto_net.generate(*kwargs)
        return res
    

class AutoNetForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelForMaskedLM.from_pretrained(config['auto_net_path'], config=self.auto_net_config)
    
    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    

class AutoNetForSequenceClassification(nn.Module):
    def __init__(self, config):
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelForSequenceClassification.from_pretrained(config['auto_net_path'], config=self.auto_net_config)

    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
    
class AutoNetForLMHead(nn.Module):
    def __init__(self, config):
        self.config = config
        self.auto_net_config = AutoConfig.from_pretrained(config['auto_net_path'], **config['auto_net_config'])
        self.auto_net = AutoModelWithLMHead.from_pretrained(config['auto_net_path'], config=self.auto_net_config)

    def forward(self, *args, **kwargs):
        res = self.auto_net(**kwargs)
        return res
"""
date: 2024/1/27
author: Zijian Wang
Describe: 自定义损失函数模块。包含基础损失函数类和由它派生的其他损失函数类，用于处理不同类型的损失计算。
"""
# from torch.nn import CrossEntropyLoss, MSELoss, TripletMarginLoss, MarginRankingLoss, BCELoss, CosineEmbeddingLoss
import torch
from torch import  nn
from torch.autograd import Variable
from transformers.trainer_pt_utils import LabelSmoother
from torch.nn import CrossEntropyLoss as CrossEntropyLossTorch, BCELoss as BCELossTorch, MSELoss as MSELossTorch

class BaseLoss():
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        自定义损失函数的基类。

        Args:
            config (Dict[str, Any]): 损失函数的配置字典。
        """
        self.config = config
        self.loss = None
    
    def __call__(self, *args, **kwargs) -> None:
        """
        损失函数调用方法的占位符。该方法应该被派生类覆盖。
        
        Raises:
            NotImplementedError: 如果派生类没有实现这个方法
        """
        raise NotImplementedError('function __call__ of BaseLoss is not implemented')

class CrossEntropyLoss(BaseLoss):
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        交叉熵损失：CrossEntropyLoss类。

        Args:
            config (Dict[str, Any]): 损失函数的配置字典。
        """
        super().__init__()
        self.loss = CrossEntropyLossTorch(**config)

    def __call__(self, *args, **kwargs) -> float:
        """
        计算并返回交叉熵损失。
        
        Args:
            pred (Tensor): 模型预测值。
            label (Tensor): 实际标签。

        Return:
            self.loss(float): 交叉熵损失值。
        """
        pred = kwargs['pred']
        label = kwargs['label']
        return self.loss(pred, label)

class BCELoss(BaseLoss):
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        处理二分类交叉熵损失的BCELoss类。

        Args:
            config (Dict[str, Any]): 损失函数的配置字典。
        """
        super().__init__(config)
        self.loss = BCELossTorch()

    def __call__(self, *args, **kwargs) -> float:
        """
        计算并返回二分类交叉熵损失。

        Args:
            pred (Tensor): 模型预测值。
            label (Tensor): 实际标签。

        Return:
            self.loss(float): 二分类交叉熵损失值。
        """
        pred = kwargs['pred']
        label = kwargs['label']
        return self.loss(pred, label)

class MSELoss(BaseLoss):
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        处理均方误差损失的MSELoss类。

        Args:
            config (Dict[str, Any]): 损失函数的配置字典。
        """
        super().__init__(config)
        self.loss = MSELossTorch()

    def __call__(self, *args, **kwargs) -> float:
        """
        计算并返回均方误差损失。

        Args:
            pred (Tensor): 模型预测值。
            label (Tensor): 实际标签。

        Return:
            self.loss(float): 均方误差损失值。
        """
        label_tran = lambda x: x.to(torch.float32)
        pred = kwargs['pred']
        label = kwargs['label']
        label = label_tran(label)
        return self.loss(pred, label)


class OutLoss(BaseLoss):
    def __init__(self, config):
        """
        输出损失的OutLoss类。

        Args:
            config (Dict[str, Any]): 损失函数的配置字典。
        """
        super().__init__(config)
    
    def __call__(self, *args, **kwargs):
        """
        输出损失值。

        Args:
            loss (float): 损失值。

        Returns:
            loss (float): 损失值。
        """
        loss = kwargs['loss']
        return loss

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


class LabelSmootherLoss(BaseLoss):
    def __init__(self, config):
        super().__init__(config)
        self.label_smoothing_factor = config['label_smoothing_factor']
        self.label_smoother = LabelSmoother(epsilon=self.config['label_smoothing_factor'])

    def __call__(self, *args, **kwargs):
        pred = kwargs['pred']
        label = kwargs['label']
        loss = self.label_smoother(pred, label)
        return loss


# class BaseLoss:
#     def __init__(self, config):
#         self.config = config
#         self.loss_name = config['loss_name']
#         config.pop('loss_name')
#         if self.loss_name == 'CrossEntropyLoss':
#             self.loss = CrossEntropyLoss(**config)
#         elif self.loss_name == 'CrossEntropyLossWeighted':
#             device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#             weight = torch.Tensor(config['weight']).to(device=device)
#             self.loss = CrossEntropyLoss(weight)
#         elif self.loss_name == 'MSELoss':
#             self.label_tran = lambda x: x.to(torch.float32)
#             self.loss = MSELoss()
#         elif self.loss_name == 'BCELoss':
#             self.loss = BCELoss()
#         elif self.loss_name == 'LabelSmootherLoss':
#             self.loss = LabelSmootherLoss(config)
#         elif self.loss_name == 'CscLoss':
#             self.loss = CscLoss(config)
#         elif self.loss_name == 'OutLoss':
#             self.loss = OutLoss(config)

#     def __call__(self, *args, **kwargs):
#         assert self.loss is not None
#         if self.loss_name == "MSELoss":
#             kwargs['label'] = self.label_tran(kwargs['label'])
#         else:
#             pred = kwargs['pred']
#             label = kwargs['label']
#             return self.loss(pred, label)

# class LabelSmootherLoss(object):
#     def __init__(self, config):
#         self.config = config
#         self.label_smoothing_factor = config['label_smoothing_factor']
#         self.label_smoother = LabelSmoother(epsilon=self.config['label_smoothing_factor'])

#     def __call__(self, pred, label):
#         loss = self.label_smoother(pred, label)
#         return loss


class FocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss.
    copy from https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, config):
        super(FocalLoss, self).__init__()
        self.config = config
        self.num_labels = config['num_labels']
        self.activation_type = config.get('activation_type', 'softmax')
        self.gamma = config.get('gamma', 2.0)
        self.alpha = config.get('alpha', 0.25)
        self.epsilon = config.get('epsilon', 1.e-9)

    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(input, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()


# class CscLoss:
#     def __init__(self, config):
#         self.config = config
#         self.loss_weight = config['loss_weight']
#         self.focal_loss = FocalLoss(**config['focal_loss'])

#     def __call__(self, pred, label):
#         word_loss = pred['word_loss']
#         det_prob = pred['det_prob']
#         det_label = label['det_label']
#         det_loss = self.focal_loss(det_prob, det_label)
#         loss = self.loss_weight * det_loss + (1 - self.loss_weight) * word_loss
#         return loss

# class OutLoss:
#     def __init__(self, config):
#         self.config = config

#     def __call__(self, pred, label):
#         loss = pred['loss']
#         return loss
        
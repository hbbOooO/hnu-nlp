"""
date: 2024/1/27
author: Mingjie Han
Describe: 基础模型类。规定所有模型的必须实现的接口。
"""
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, config):
        """初始化工作，设置配置参数

        Args:
            config:配置参数

        """
        super().__init__()
        self.config = config

    def forward(self, batch):
        """前向传播方法

        Args:
            batch: 批次化

        """
        if self.training:
            return self.forward_train_base(batch)
        else:
            return self.forward_test_base(batch)

    def forward_train_base(self, batch):
        """训练模式下的基础前向传播方法

        Args:
            batch: 批次化

        """
        self.forward_both(batch)
        return self.forward_train(batch)

    def forward_test_base(self, batch):
        """测试模式下的基础前向传播方法

        Args:
            batch: 批次化

        """
        self.forward_both(batch)
        return self.forward_test(batch)

    def forward_both(self, batch):
        """共同部分的前向传播方法

        Args:
            batch: 批次化

        """
        # 抛出未实现错误，要求子类在自己的实现中重写这个方法
        raise NotImplementedError('forward_both Function is not Implemented in son class of BaseModel')

    def forward_train(self, batch):
        """训练模式下的的前向传播方法

        Args:
            batch: 批次化

        """
        # 抛出未实现错误，要求子类在自己的实现中重写这个方法
        raise NotImplementedError('forward_train Function is not Implemented in son class of BaseModel')
    
    def forward_test(self, batch):
        """训练模式下的的前向传播方法

        Args:
            batch: 批次化

        """
        # 抛出未实现错误，要求子类在自己的实现中重写这个方法
        # raise NotImplementedError('forward_test Function is not Implemented in son class of BaseModel')
        # 若子类未重写该方法，默认调用forward_train(batch)方法
        self.forward_train(batch)
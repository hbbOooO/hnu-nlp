"""
date: 2024/1/21
author: Mingjie Han
Describe: 基础数据集类。规定所有数据集的必须实现的接口。
"""
from typing import Dict, Any, List
from torch.utils.data.dataset import Dataset

class BaseDataset(Dataset):
    def __init__(self, config: Dict[str, Any], processors: List[object]) -> None:
        """初始化工作，包括设置数据路径、数据集类型、设置数据处理管道

        Args:
            config (Dict[str, Any]): 数据集配置字典
            processors (List[object]): 数据处理对象列表
        """
        super(BaseDataset, self).__init__()
        self.config = config
        self.dataset_type = config['dataset_type']
        assert self.dataset_type == 'train' or self.dataset_type == 'val' or self.dataset_type == 'inference'
        self.data_root_dir = config['data_root_dir'] + '/' if 'data_root_dir' in config else ''
        self.data_paths = [
            self.data_root_dir + path for path in config['data_file_path']
        ]
        self.data = self._read(self.data_paths)
        self.data = self._add_id(self.data)
        if self.dataset_type != 'inference': self.gt_label = self._convert_gt_label()
        self.processors = processors

    def _read(self, data_paths: List[str]) -> List[Dict[str, Any]]:
        """从本地中读取数据

        Args:
            data_paths (List[str]):  本地数据路径

        Reutrn:
            data (List[Dict[str, Any]]):  所用数据
        """
        raise NotImplementedError('_read Function is not Implemented in son class of BaseDataset')
    
    def _add_id(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为每一项数据设置唯一的id
        
        Args:
            data (List[Dict[str, Any]]): 所用数据
        """
        for i in range(len(data)):
            data[i]['id'] = i
        return data

    def _convert_gt_label(self, data: List[Dict[str, Any]]) -> Dict[int, Any]:
        """得到GT标签（对于训练集和测试集而言）

        Args:
            data (List[Dict[str, Any]]):  所用数据

        Return:
            gt_label Dict[int, Any]:  标签数据
        """
        raise NotImplementedError('_convert_gt_label Function is not Implemented in son class of BaseDataset')
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """覆盖Datset基类的__getitem__方法
        
        Args:
            index (int): Dataloader中的编号，与上面_add_id()方法中的id不一样

        Return:
            sample (Dict[str, Any]): 用于批次化(batch)的数据
        """
        item = self.data[index]
        sample = {}
        for name, processor in self.processors.items():
            processor(item, sample)
        return sample

    def __len__(self) -> int:
        """覆盖Dataset基类的__len__方法

        Return:
            length of data (int): 数据数量
        
        """
        return len(self.data)
    
    def evaluate(self, prediction: Dict[str, Any]) -> Dict(str, Any):
        """评估预测的结果

        Args:
            prediction (Dict[str, Any])  预测数据

        Return:
            metric (Dict[str, Any]) 评估结果
        
        """
        raise NotImplementedError('evaluate Function is not Implemented in son class of BaseDataset')
    
    def out(self, prediction: Dict[str, Any], path: str) -> None:
        """将预测结果输出到本地文件
        
        Args:
            prediction (Dict[str, Any]):  预测数据
            path (str):  输出文件地址

        """
        raise NotImplementedError('out Function is not Implemented in son class of BaseDataset')
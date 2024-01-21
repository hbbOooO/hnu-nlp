from torch.utils.data.dataset import Dataset

class BaseDataset(Dataset):
    def __init__(self, config, processors):
        super(BaseDataset, self).__init__()
        self.config = config
        self.dataset_type = config['dataset_type']
        assert self.dataset_type == 'train' or self.dataset_type == 'val' or self.dataset_type == 'inference'
        self.data_root_dir = config['data_root_dir'] + '/' if 'data_root_dir' in config else ''
        self.data_paths = [
            self.data_root_dir + path for path in config['data_file_path']
        ]
        self._read()
        self._add_id()
        if self.dataset_type != 'inference': self._convert_gt_label()
        self.processors = processors

    def _read(self):
        """从本地中读取数据的方法

        输入: 
        
        self.data_paths: List(str)  本地数据路径

        输出:

        self.data: List(Dict())  所用数据
        """
        raise NotImplementedError('_read Function is not Implemented in son class of BaseDataset')
    
    def _add_id(self):
        """数据集中加上特有的id号码
         
        """
        data = self.data
        for i in range(len(data)):
            data[i]['id'] = i
        self.data = data

    def _convert_gt_label(self):
        """得到数据中的标签

        输入:

        self.data: List(Dict())  所用数据

        输出
        
        self.gt_label: Dict()  标签数据
        """
        raise NotImplementedError('_convert_gt_label Function is not Implemented in son class of BaseDataset')
    
    def __getitem__(self, index):
        item = self.data[index]
        sample = {}
        for name, processor in self.processors.items():
            processor(item, sample)
        return sample

    def __len__(self):
        return len(self.data)
    
    def evaluate(self, prediction):
        """评估预测的结果

        输入
        
        self.gt_label: Dict()  标签数据

        @param prediction: Dict()  预测数据

        输出

        @return metric: Dict() 评估结果
        
        """
        raise NotImplementedError('evaluate Function is not Implemented in son class of BaseDataset')
    
    def out(self, prediction, path):
        """将预测结果打印到本地
        
        输入

        @param prediction: Dict()  预测数据

        @param path: str  打印文件地址

        输出

        无

        """
        raise NotImplementedError('out Function is not Implemented in son class of BaseDataset')
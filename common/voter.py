from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm
import numpy as np

from .losses import BaseLoss
from .logger import Logger
from .timer import Timer
from .checkpoint import CheckPoint
from .optimizer import Optimizer

class Voter:
    def __init__(self, config):
        self.config = config
        self.dataset_config = config['dataset']
        self.model_config = config['model']
        self.run_param = config['run_param']
        self.train_param = self.run_param['train_param']
        self.val_param = self.run_param['val_param']
        self.inference_param = self.run_param['inference_param']
        self.loss_param = self.train_param['loss']
        self.optimizer_param = self.train_param['optimizer']
        self.resume_file = self.run_param.get('resume_file', None)

        Logger.get_logger().info('the config is as following: \n' + str(self.config))
        
        self._init_extra()
        self._import_classes()
        self._init_dataset()
        self._init_model()
        self._init_dataloader()
        # self._load_resume()
        

    def _import_classes(self):
        Logger.get_logger().info('----- import dataset class -----')
        dataset_name = self.dataset_config['class_name']
        dataset_module_path = self.dataset_config['module_path']
        dataset_module = __import__(dataset_module_path, fromlist=[dataset_module_path.split('.')[-1]])
        dataset_class = getattr(dataset_module, dataset_name)
        assert dataset_class
        self.dataset_class = dataset_class

        Logger.get_logger().info('----- import model class -----')
        model_name = self.model_config['class_name']
        model_module_path = self.model_config['module_path']
        model_module = __import__(model_module_path, fromlist=[model_module_path.split('.')[-1]])
        model_class = getattr(model_module, model_name)
        assert model_class
        self.model_class = model_class

    def _init_dataset(self):
        Logger.get_logger().info('----- init dataset -----')
        dataset_configs = self.dataset_config['datasets']
        datasets = {}
        for dataset_config in dataset_configs:
            datasets[dataset_config['dataset_type']] = self.dataset_class(dataset_config)
        self.datasets = datasets

    def _init_model(self):
        Logger.get_logger().info('----- init model -----')
        model_config = self.model_config['model']
        models = []
        for path in self.resume_file:
            model = self.model_class(model_config)
            ckpt = self.ckpter.load(path)
            model.load_state_dict(ckpt['model'])
            models.append(model)
        self.models = models

        Logger.get_logger().info(model)

    def _init_dataloader(self):
        Logger.get_logger().info('----- init dataloader -----')
        dataloaders = dict()
        for dataset_type, dataset in self.datasets.items():
            if dataset_type == 'train': batch_size = self.train_param['batch_size']
            elif dataset_type == 'val': batch_size = self.val_param['batch_size']
            elif dataset_type == 'inference': batch_size = self.inference_param['batch_size']
            dataloaders[dataset_type] = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True)
        self.dataloaders = dataloaders

    # def _init_optimizer(self):
    #     Logger.get_logger().info('----- init optimizer -----')
    #     optimizer_scheduler = Optimizer(self.optimizer_param, self.model.parameters())
    #     self.optimizer = optimizer_scheduler.optimizer
    #     self.scheduler = optimizer_scheduler.scheduler

    # def _init_loss(self):
    #     Logger.get_logger().info('----- init loss -----')
    #     self.criterion = BaseLoss(self.loss_param)
    
    # def _init_meter(self):
    #     Logger.get_logger().info('----- import meter class -----')
    #     self.meter_config = self.run_param['meter']
    #     meter_name = self.meter_config['class_name']
    #     meter_module_path = self.meter_config['module_path']
    #     meter_module = __import__(meter_module_path, fromlist=[meter_module_path.split('.')[-1]])
    #     meter_class = getattr(meter_module, meter_name)
    #     self.METER_CLASS = meter_class

    

    def _init_extra(self):
        Logger.get_logger().info('----- init extra -----')
        # only apply one GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        Timer.set_up(self.run_param['timer_type'])
        self.ckpter = CheckPoint(self.train_param['checkpoint'])
        self.resume_epoch = getattr(self, 'resume_epoch', None)



    def _to_cuda(self, batch):
        device = self.device
        if isinstance(batch, list):
            for i in range(len(batch)): batch[i] = self._to_cuda(batch[i])
        if isinstance(batch, dict):
            fields = batch.keys()
            for field in fields:
                batch[field] = self._to_cuda(batch[field])
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        return batch

    def __call__(self):
        train_type = self.run_param['run_type']
        if 'train' in train_type:
            self.train()
        if 'val' in train_type:
            self.val()
        if 'inference' in train_type:
            self.inference()

    def train(self):
        NotImplementedError


    def val(self):
        NotImplementedError

    def inference(self):
        self.inference_prediction = {}
        with torch.no_grad():
            for model in self.models:
                model.to(self.device)
                model.eval()
            for batch in tqdm(self.dataloaders['inference']):
                prepared_batch = self._to_cuda(batch)
                pred_w_label_list = []
                for model in self.models:
                    _, pred_w_label = model(prepared_batch)
                    pred_w_label = pred_w_label.cpu().numpy()
                    pred_w_label_list.append(pred_w_label)
                voted_pred_w_label = np.zeros(pred_w_label_list[0].shape,dtype=int)
                voted_pred_w_label[0][0] = pred_w_label_list[0][0][0]
                for i in range(1, len(voted_pred_w_label[0])):
                    select_num = len([item for item in pred_w_label_list if item[0][i]==1])
                    not_select_num = len([item for item in pred_w_label_list if item[0][i]==0])
                    if select_num > not_select_num:
                        voted_pred_w_label[0][i] = 1
                    else:
                        voted_pred_w_label[0][i] = 0
                self._update_inference_meter(voted_pred_w_label)
            # print(self.inference_prediction)
            self._inference_out()
            


    def _update_inference_meter(self, pred_w_label):
        # update val results
        pred_w_label = pred_w_label
        pred_w_label = {item[0]: item[1:] for item in pred_w_label}
        self.inference_prediction.update(pred_w_label)

    def _inference_out(self):
        self.datasets['inference'].out(self.inference_prediction, self.inference_param['out_path'])









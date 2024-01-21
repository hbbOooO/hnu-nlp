from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm
import json
import os
import random
import numpy as np

from .logger import Logger
from .timer import Timer
from .checkpoint import CheckPoint
from .optimizer import Optimizer
from .ema import EMA
from .fgm import FGM

class Trainer:

    # COLLATE_PATH = 'common.collate'

    def __init__(self, config, args):
        self.config = config
        self.args = args

        Logger.get_logger().info('the argparse is as following: \n' + json.dumps(vars(self.args), indent=2, separators=(', ', ': '), ensure_ascii=False))
        Logger.get_logger().info('the config is as following: \n' + json.dumps(self.config, indent=2, separators=(', ', ': '), ensure_ascii=False))

        self.processor_configs = config['processors']
        self.dataset_config = config['dataset']
        self.model_config = config['model']
        self.run_param = config['run_param']
        self._init_run_param()
        self.resume_file = self.run_param.get('resume_file', None)

        self._set_seed()
        self._import_classes()
        self._init_processors()
        self._init_dataset()
        self._init_model()
        self._init_dataloader()
        self._init_optimizer()
        self._init_loss()
        self._init_extra()
        if self.resume_file is not None: self._load_resume()

    def _init_run_param(self):
        train_type = self.run_param['run_type']
        self.is_train = train_type == 'train'
        self.is_val = train_type == 'val' or (self.is_train and self.run_param.get('train_param', False) and self.run_param['train_param'].get('val_on_val_set', False))
        self.is_inference = train_type == 'inference'

        dataset_types = [item['dataset_type'] for type, item in self.dataset_config['datasets'].items()]
        
        self.use_ema = False
        self.use_fgm = False
        if self.is_train:
            assert 'train_param' in self.run_param
            self.train_param = self.run_param['train_param']
            self.loss_param = self.train_param['loss']
            self.optimizer_param = self.train_param['optimizer']
            assert 'train' in dataset_types
            if self.train_param.get('val_on_val_set', False):
                assert 'val' in dataset_types and 'val_param' in self.run_param
                self.val_param = self.run_param['val_param']
            self.use_ema = 'ema' in self.train_param
            self.use_fgm = 'fgm' in self.train_param
        if self.is_val:
            assert 'val_param' in self.run_param
            assert 'val' in dataset_types
            self.val_param = self.run_param['val_param']
        if self.is_inference:
            assert 'inference_param' in self.run_param
            assert 'inference' in dataset_types
            self.inference_param = self.run_param['inference_param']

        dataloader_types = list(self.run_param['dataloader'].keys())
        assert len(dataloader_types) != 0
        self.dataloader_config_dict = {}
        if len(dataloader_types) == 1:
            if self.is_train:
                self.dataloader_config_dict['train'] = self.run_param['dataloader'][dataloader_types[0]].copy()
            if self.is_val:
                self.dataloader_config_dict['val'] = self.run_param['dataloader'][dataloader_types[0]].copy()
            if self.is_inference:
                self.dataloader_config_dict['inference'] = self.run_param['dataloader'][dataloader_types[0]].copy()
        else:
            for key, value in self.run_param['dataloader'].items():
                self.dataloader_config_dict[key] = value

        # dataLoader_types = [item['dataloader_type'] for item in self.run_param['dataloader']]
        # assert len(dataLoader_types) != 0
        # self.dataloader_config_dict = {}
        # if len(dataLoader_types) == 1:
        #     if self.is_train:
        #         self.dataloader_config_dict['train'] = self.run_param['dataloader'][0]['config'].copy()
        #     if self.is_val:
        #         self.dataloader_config_dict['val'] = self.run_param['dataloader'][0]['config'].copy()
        #     if self.is_inference:
        #         self.dataloader_config_dict['inference'] = self.run_param['dataloader'][0]['config'].copy()
        # else:
        #     for item in self.run_param['dataloader']:
        #         dataloader_type = item['dataloader_type']
        #         self.dataloader_config_dict[dataloader_type] = item['config']

    def _set_seed(self):
        seed = self.config['run_param'].get('seed', None)
        if seed is not None:
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


    def _import_classes(self):
        Logger.get_logger().info('----- import processor classes -----')
        processor_class_dict = {}
        for processor_id, processor_config in self.processor_configs.items():
            processor_name = processor_config['class_name']
            processor_module_path = processor_config['module_path']
            processor_module =  __import__(processor_module_path, fromlist=[processor_module_path.split('.')[-1]])
            processor_class = getattr(processor_module, processor_name)
            processor_class_dict[processor_id] = (processor_class, processor_config['config'])
        assert processor_class_dict
        self.processor_class_dict = processor_class_dict

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

        Logger.get_logger().info('----- import meter class -----')
        self.meter_config = self.run_param['meter']
        meter_name = self.meter_config['class_name']
        meter_module_path = self.meter_config['module_path']
        meter_module = __import__(meter_module_path, fromlist=[meter_module_path.split('.')[-1]])
        meter_class = getattr(meter_module, meter_name)
        self.METER_CLASS = meter_class

        if self.is_train:
            Logger.get_logger().info('----- import loss class -----')
            loss_name = self.loss_param['class_name']
            loss_module_path = self.loss_param['module_path']
            loss_module = __import__(loss_module_path, fromlist=[loss_module_path.split('.')[-1]])
            loss_class = getattr(loss_module, loss_name)
            assert loss_class
            self.loss_class = loss_class
            
    
    def _init_processors(self):
        Logger.get_logger().info('----- init processors -----')
        processor_class_dict = self.processor_class_dict
        self.processor_dict = {}
        for id, value in processor_class_dict.items():
            processor_class, processor_config = value
            self.processor_dict[id] = processor_class(processor_config)

    def _init_dataset(self):
        Logger.get_logger().info('----- init dataset -----')
        dataset_configs = self.dataset_config['datasets']
        datasets = {}
        for type, dataset_config in dataset_configs.items():
            processor_ids = dataset_config['processor_ids']
            # processors = [self.processor_dict[id] for id in processor_ids]
            processors = {processor_id: processor for processor_id, processor in self.processor_dict.items() if processor_id in processor_ids}
            datasets[dataset_config['dataset_type']] = self.dataset_class(dataset_config, processors)
        self.datasets = datasets

        # dataset_configs = self.dataset_config['datasets']
        # datasets = {}
        # for dataset_config in dataset_configs:
        #     processor_ids = dataset_config['processor_ids']
        #     processors = [self.processor_dict[id] for id in processor_ids]
        #     datasets[dataset_config['dataset_type']] = self.dataset_class(dataset_config, processors)
        # self.datasets = datasets

    def _init_model(self):
        Logger.get_logger().info('----- init model -----')
        model_config = self.model_config['config']
        model = self.model_class(model_config)
        self.model = model
        

    def _init_dataloader(self):
        Logger.get_logger().info('----- init dataloader -----')

        dataloaders = dict()
        for run_type in ['train', 'val', 'inference']:
            if run_type == 'train' and self.is_train or \
                run_type == 'val' and self.is_val or \
                run_type == 'inference' and self.is_inference:
                dataloader_config = self.dataloader_config_dict[run_type]
                if 'collate_fn' in dataloader_config:
                    Logger.get_logger().info('----- import {} collate function -----'.format(run_type))
                    collate_fn_config = dataloader_config['collate_fn']
                    collate_name = collate_fn_config['fn_name']
                    collate_module_path = collate_fn_config['module_path']
                    collate_module = __import__(collate_module_path, fromlist=[collate_module_path.split('.')[-1]])
                    collate_fn = getattr(collate_module, collate_name)
                    assert collate_fn
                    dataloader_config['collate_fn'] = collate_fn
                dataloaders[run_type] = DataLoader(dataset=self.datasets[run_type], **dataloader_config)
        self.dataloaders = dataloaders

    def _init_optimizer(self):
        if self.is_train:
            Logger.get_logger().info('----- init optimizer -----')
            optimizer_scheduler = Optimizer(self.optimizer_param, self.model.parameters())
            self.optimizer = optimizer_scheduler.optimizer
            self.scheduler = optimizer_scheduler.scheduler
            self.gradient_accumulation_steps = self.optimizer_param.get('gradient_accumulation_steps', 1)

    def _init_loss(self):
        if self.is_train:
            Logger.get_logger().info('----- init loss -----')
            self.criterion = self.loss_class(self.loss_param.get('config', {}))
        

    def _load_resume(self):
        Logger.get_logger().info('----- load resume -----')
        ckpt = torch.load(self.resume_file, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if self.is_train:
            self.resume_from_start = self.train_param.get('resume_from_start', False)
            if not self.resume_from_start:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.resume_config = ckpt['run_config']
                self.resume_epoch = ckpt['epoch']
                self.val_metric = ckpt.get('val_metric', None)
                self.best_metric_epoch = ckpt.get('best_epoch', None)
                self.best_metric = ckpt.get('best_val_metric', None)
                
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device=self.device)

    def _init_extra(self):
        Logger.get_logger().info('----- init extra -----')
        # only apply one GPU
        cuda_idx = str(self.run_param.get('cuda_idx', 0))
        self.device = torch.device("cuda:" + cuda_idx if torch.cuda.is_available() and cuda_idx != '-1' else "cpu")
        Timer.set_up(self.run_param['timer_type'])
        if self.is_train:
            self.ckpter = CheckPoint(self.train_param['checkpoint'], self.train_param.get('save_module', None))
            self.val_on_train_set = self.train_param.get('val_on_train_set', False)
            self.val_on_val_set = self.train_param.get('val_on_val_set', False)
            self.resume_epoch = getattr(self, 'resume_epoch', None)
            # ema
            if self.use_ema:
                self.ema_config = self.train_param['ema'].get('config', {})
                self.model.to(self.device)
                self.ema_model = EMA(self.model, **self.ema_config)
                self.ema_model.register()
                # self.ema_model.model.to(self.device)
            if self.use_fgm:
                self.fgm_config = self.train_param['fgm'].get('config', {})
                self.fgm = FGM(self.model, **self.fgm_config)


        

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
        Logger.get_logger().info('start training')
        self.max_epoch = self.train_param['max_epoch']
        self.max_iteration = len(self.dataloaders['train'])
        self.curr_epoch = 0 if self.resume_file is None or self.resume_epoch is None else self.resume_epoch

        if self.val_on_val_set:
            self.epoch_val_metric = []
            self.best_metric_epoch = 0
            self.best_metric = self.METER_CLASS()
        
        self.model.to(self.device)
        self.model.train()

        for epoch_index in range(self.max_epoch-self.curr_epoch):
            self.curr_epoch += 1
            self.curr_iteration = 0

            self.train_prediction = {}
            self.train_losses = []

            for batch in self.dataloaders['train']:
                self.curr_iteration += 1
                prepared_batch = self._to_cuda(batch)
                loss_input, pred_w_index = self.model(prepared_batch)
                loss = self.criterion(**loss_input)
                self._update_train_meter(pred_w_index, loss)
                self._backward(loss, prepared_batch)
                self._report()
                # test
                # self._epoch_summary()

            self._epoch_summary()
            

    def _get_pred_dict(self, pred_w_index):
        pred_dict = {}
        ids = None
        for key, value in pred_w_index.items():
            value = value.cpu().numpy()
            if key == 'id':
                ids = value
            else:
                for i in range(len(value)):
                    id = ids[i]
                    item = value[i]
                    if id not in pred_dict:
                        pred_dict[id] = {
                            key: item
                        }
                    else:
                        pred_dict[id][key] = item
        return pred_dict

                
    def _update_train_meter(self, pred_w_index, loss):
        # update train prediction
        if self.val_on_train_set:
            pred_dict = self._get_pred_dict(pred_w_index)
            self.train_prediction.update(pred_dict)
        self.train_losses.append(loss.item())


    def _backward(self, loss, prepared_batch):
        # self.optimizer.zero_grad()
        loss.backward()
        if self.use_fgm:
            self.fgm.attack()
            fgm_loss_input, fgm_pred_w_index = self.model(prepared_batch)
            fgm_loss = self.criterion(**fgm_loss_input)
            fgm_loss.backward()
            self.fgm.restore()

        self.scheduler.step((self.curr_epoch,self.curr_iteration,self.max_epoch,self.max_iteration))

        if self.curr_iteration % self.gradient_accumulation_steps == 0 or self.curr_iteration == self.max_iteration:
            self.optimizer.step()    
            self.optimizer.zero_grad()
        if self.use_ema:
            self.ema_model.update()

    def _report(self):
        log_interval = self.train_param['log_interval']
        if self.curr_iteration % log_interval == 0:
            # update metric
            if self.val_on_train_set:
                metric = self.METER_CLASS(self.datasets['train'].evaluate(self.train_prediction))
                self.train_prediction.clear()
            else:
                metric = None

            Logger.get_logger().info(
                'epoch: {}/{}, iteration: {}/{}, loss(avg): {:6f}({:.6f}), train metric:{}, lr: {:.2g}, cost time: {}, reamin time: {}'.format(
                    self.curr_epoch, self.max_epoch, 
                    self.curr_iteration, self.max_iteration,
                    self.train_losses[-1], sum(self.train_losses)/len(self.train_losses),
                    metric,
                    self.optimizer.state_dict()['param_groups'][0]['lr'],
                    Timer.calculate_spend(),
                    Timer.calculate_remain(self.curr_epoch, self.curr_iteration, self.max_epoch, self.max_iteration, resume_epoch=self.resume_epoch)
                )
            )
    
    def _epoch_summary(self):
        if self.use_ema:
            self.ema_model.apply_shadow()

        if self.val_on_val_set:
            # evaluate on val dataset
            self.val()
            self.model.train()
            # compare val result with previous
            self.epoch_val_metric.append(self.val_metric)
            if self.val_metric > self.best_metric:
                self.best_metric = self.val_metric
                self.best_metric_epoch = self.curr_epoch
        else:
            self.val_metric = None
            self.best_metric_epoch = None
            self.best_metric = None

        # save_model = self.model if not self.use_ema else self.ema_model

        # save checkpoint
        self.ckpter.save(self.curr_epoch, self.model, self.optimizer, self.config, self.val_on_val_set, val_metric=self.val_metric, best_epoch=self.best_metric_epoch, best_val_metric=self.best_metric)
        # save module of moel
        self.ckpter.save_module(self.curr_epoch, self.model, best_epoch=self.best_metric_epoch)

        if self.use_ema:
            self.ema_model.restore()
        
        # report
        Logger.get_logger().info('Epoch {} finished. train loss: {:.4f}, val metric:{}, bset epoch: {}, best metric: {}, cost time: {}, reamin time: {}'.format(
            self.curr_epoch,
            sum(self.train_losses)/len(self.train_losses),
            self.val_metric,
            self.best_metric_epoch,
            self.best_metric,
            Timer.calculate_spend(),
            Timer.calculate_remain(self.curr_epoch, self.curr_iteration, self.max_epoch, self.max_iteration, resume_epoch=self.resume_epoch)
        ))


    def val(self):
        self.val_prediction = {}

        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            for batch in tqdm(self.dataloaders['val']):
                prepared_batch = self._to_cuda(batch)
                _, pred_w_index = self.model(prepared_batch)
                self._update_val_meter(pred_w_index)
            metric = self.datasets['val'].evaluate(self.val_prediction)
            self.val_metric = self.METER_CLASS(metric)
            Logger.get_logger().info('full result in val: {}'.format(
                self.val_metric
            ))
        
        
            
            

    def _update_val_meter(self, pred_w_index):
        # update val results
        pred_dict = self._get_pred_dict(pred_w_index)
        self.val_prediction.update(pred_dict)


    def inference(self):
        self.inference_prediction = {}
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            for batch in tqdm(self.dataloaders['inference']):
                prepared_batch = self._to_cuda(batch)
                _, pred_w_index = self.model(prepared_batch)
                self._update_inference_meter(pred_w_index)
            self._inference_out()
            


    def _update_inference_meter(self, pred_w_index):
        # update val results
        pred_dict = self._get_pred_dict(pred_w_index)
        self.inference_prediction.update(pred_dict)

    def _inference_out(self):
        self.datasets['inference'].out(self.inference_prediction, self.inference_param['out_path'])









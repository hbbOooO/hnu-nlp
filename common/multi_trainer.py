from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data.distributed import DistributedSampler

from .losses import BaseLoss
from .logger import Logger
from .timer import Timer
from .checkpoint import CheckPoint
from .optimizer import Optimizer

class MultiTrainer:
    def __init__(self, config, args):
        self.config = config
        self.local_rank = args.local_rank
        self.dataset_config = config['dataset']
        self.model_config = config['model']
        self.run_param = config['run_param']
        self.train_param = self.run_param['train_param']
        self.val_param = self.run_param['val_param']
        self.inference_param = self.run_param['inference_param']
        self.loss_param = self.train_param['loss']
        self.optimizer_param = self.train_param['optimizer']
        self.resume_file = self.run_param.get('resume_file', None)

        if self.local_rank == 0: Logger.get_logger().info('the config is as following: \n' + str(self.config))

        self._import_classes()
        self._init_dataset()
        self._init_model()
        self._init_dataloader()
        self._init_optimizer()
        self._init_loss()
        self._init_meter()
        self._init_extra()
        if self.resume_file is not None: self._load_resume()
        self._init_distributed()
        

    def _import_classes(self):
        if self.local_rank == 0: Logger.get_logger().info('----- import dataset class -----')
        dataset_name = self.dataset_config['class_name']
        dataset_module_path = self.dataset_config['module_path']
        dataset_module = __import__(dataset_module_path, fromlist=[dataset_module_path.split('.')[-1]])
        dataset_class = getattr(dataset_module, dataset_name)
        assert dataset_class
        self.dataset_class = dataset_class

        if self.local_rank == 0: Logger.get_logger().info('----- import model class -----')
        model_name = self.model_config['class_name']
        model_module_path = self.model_config['module_path']
        model_module = __import__(model_module_path, fromlist=[model_module_path.split('.')[-1]])
        model_class = getattr(model_module, model_name)
        assert model_class
        self.model_class = model_class

    def _init_dataset(self):
        if self.local_rank == 0: Logger.get_logger().info('----- init dataset -----')
        dataset_configs = self.dataset_config['datasets']
        datasets = {}
        for dataset_config in dataset_configs:
            datasets[dataset_config['dataset_type']] = self.dataset_class(dataset_config)
        self.datasets = datasets

    def _init_model(self):
        if self.local_rank == 0: Logger.get_logger().info('----- init model -----')
        model_config = self.model_config['model']
        model = self.model_class(model_config)
        self.model = model
        if self.local_rank == 0: Logger.get_logger().info(model)

    def _init_dataloader(self):
        if self.local_rank == 0: Logger.get_logger().info('----- init dataloader -----')
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

    def _init_optimizer(self):
        if self.local_rank == 0: Logger.get_logger().info('----- init optimizer -----')
        optimizer_scheduler = Optimizer(self.optimizer_param, self.model.parameters())
        self.optimizer = optimizer_scheduler.optimizer
        self.scheduler = optimizer_scheduler.scheduler

    def _init_loss(self):
        if self.local_rank == 0: Logger.get_logger().info('----- init loss -----')
        self.criterion = BaseLoss(self.loss_param)
    
    def _init_meter(self):
        if self.local_rank == 0: Logger.get_logger().info('----- import meter class -----')
        self.meter_config = self.run_param['meter']
        meter_name = self.meter_config['class_name']
        meter_module_path = self.meter_config['module_path']
        meter_module = __import__(meter_module_path, fromlist=[meter_module_path.split('.')[-1]])
        meter_class = getattr(meter_module, meter_name)
        self.METER_CLASS = meter_class

    def _load_resume(self):
        if self.local_rank == 0: Logger.get_logger().info('----- load resume -----')
        ckpt = self.ckpter.load(self.resume_file)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.resume_config = ckpt['run_config']
        # self.curr_epoch = ckpt['epoch']
        self.resume_epoch = ckpt['epoch']
        self.val_metric = ckpt.get('val_metric', None)
        self.best_metric_epoch = ckpt.get('best_epoch', None)
        self.best_metric = ckpt.get('best_val_metric', None)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=self.device)

    def _init_extra(self):
        if self.local_rank == 0: Logger.get_logger().info('----- init extra -----')
        # only apply one GPU
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        Timer.set_up(self.run_param['timer_type'])
        self.ckpter = CheckPoint(self.train_param['checkpoint'])
        self.val_on_train_set = self.train_param.get('val_on_train_set', False)
        self.val_on_val_set = self.train_param.get('val_on_val_set', False)
        # Meter.set_category_num(self.run_param['meter_category_num'])
        self.resume_epoch = getattr(self, 'resume_epoch', None)

    
    def _init_distributed(self):
        if self.local_rank == 0: Logger.get_logger().info('----- init distributed -----')
        num_workers = self.run_param['num_workers']
        torch.cuda.set_device(self.local_rank)
        self.device=torch.device("cuda", self.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        self.model.to(self.device)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                output_device=self.local_rank)
        train_sampler = DistributedSampler(self.datasets['train'])
        self.dataloaders['train'] = DataLoader(self.datasets['train'], sampler=train_sampler, batch_size=self.train_param['batch_size'],
                                #   num_workers=num_workers,
                                   pin_memory=True)



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
        if self.local_rank == 0: Logger.get_logger().info('start training')
        self.max_epoch = self.train_param['max_epoch']
        self.max_iteration = len(self.dataloaders['train'])
        self.curr_epoch = 0 if self.resume_file is None else self.resume_epoch

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
            self.dataloaders['train'].sampler.set_epoch(epoch_index)
            for batch in self.dataloaders['train']:
                self.curr_iteration += 1
                prepared_batch = self._to_cuda(batch)
                loss_input, pred_w_index = self.model(prepared_batch)
                loss = self.criterion(**loss_input)
                self._update_train_meter(pred_w_index, loss)
                self._backward(loss)
                self._report()
                # test
                # self._epoch_summary()

            self._epoch_summary()
            


                
    def _update_train_meter(self, pred_w_label, loss):
        # update train prediction
        pred_w_label = pred_w_label.cpu().numpy()
        pred_w_label = {item[0]: item[1:] for item in pred_w_label}
        self.train_prediction.update(pred_w_label)
        # update train loss
        self.train_losses.append(loss.item())


    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step((self.curr_epoch,self.curr_iteration,self.max_epoch,self.max_iteration))

    def _report(self):
        log_interval = self.train_param['log_interval']
        if self.curr_iteration % log_interval == 0:
            # update metric
            if self.val_on_train_set:
                metric = self.METER_CLASS(self.datasets['train'].evaluate(self.train_prediction))
                self.train_prediction.clear()
            else:
                metric = None

            if self.local_rank == 0: Logger.get_logger().info(
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

        # save checkpoint
        self.ckpter.save(self.curr_epoch, self.model, self.optimizer, self.config, self.val_on_val_set, val_metric=self.val_metric, best_epoch=self.best_metric_epoch, best_val_metric=self.best_metric)
        # report
        if self.local_rank == 0: Logger.get_logger().info('Epoch {} finished. train loss: {:.4f}, val metric:{}, bset epoch: {}, best metric: {}, cost time: {}, reamin time: {}'.format(
            self.curr_epoch,
            sum(self.train_losses)/len(self.train_losses),
            self.val_metric,
            self.best_metric_epoch,
            self.best_metric,
            Timer.calculate_spend(),
            Timer.calculate_remain(self.curr_epoch, self.curr_iteration, self.max_epoch, self.max_iteration, resume_epoch=self.resume_epoch)
        ))
        
        # save some module of our model
        if self.train_param.get('save_module', False):
            module_save_path = self.train_param['module_save_path']
            self.model.save_module(module_save_path)




    def val(self):
        self.val_prediction = {}
        
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            for batch in tqdm(self.dataloaders['val']):
                prepared_batch = self._to_cuda(batch)
                _, pred_w_label = self.model(prepared_batch)
                self._update_val_meter(pred_w_label)
            metric = self.datasets['val'].evaluate(self.val_prediction, True)
            self.val_metric = self.METER_CLASS(metric)
            if self.local_rank == 0: Logger.get_logger().info('full result in val: {}'.format(
                self.val_metric
            ))
            
            

    def _update_val_meter(self, pred_w_label):
        # update val results
        pred_w_label = pred_w_label.cpu().numpy()
        pred_w_label = {item[0]: item[1:] for item in pred_w_label}
        self.val_prediction.update(pred_w_label)


    def inference(self):
        self.inference_prediction = {}
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            for batch in tqdm(self.dataloaders['inference']):
                prepared_batch = self._to_cuda(batch)
                _, pred_w_label = self.model(prepared_batch)
                self._update_inference_meter(pred_w_label)
            # print(self.inference_prediction)
            self._inference_out()
            


    def _update_inference_meter(self, pred_w_label):
        # update val results
        pred_w_label = pred_w_label.cpu().numpy()
        pred_w_label = {item[0]: item[1:] for item in pred_w_label}
        self.inference_prediction.update(pred_w_label)

    def _inference_out(self):
        self.datasets['inference'].out(self.inference_prediction, self.inference_param['out_path'])









from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

class Optimizer:
    def __init__(self, config, parameters):
        self.config = config
        self.use_warmup = config['use_warmup']
        if self.use_warmup:
            self.warmup_iter = config['warmup_iter']
        else:
            self.warmup_iter = None
        self.lr = config['lr']
        self.lr_strategy = config.get('lr_strategy', None)
        self.optimizer_name = config['optimizer_name']
        self.optimizer_config = config.get('optimizer_config', {})

        if self.optimizer_name == 'sgd' or self.optimizer_name == 'SGD':
            self.optimizer = SGD(
                parameters,
                lr=self.lr,
                **self.optimizer_config
            )
        elif self.optimizer_name == 'adam' or self.optimizer_name == 'Adam':
            self.optimizer = Adam(
                parameters,
                lr=self.lr,
                **self.optimizer_config
            )
        elif self.optimizer_name == 'adamw' or self.optimizer_name == 'AdamW':
            self.optimizer = AdamW(
                parameters,
                lr=self.lr,
                **self.optimizer_config  
            )

        self._init_strategy()
        
    def _init_strategy(self):
        if self.lr_strategy == 'step':
            self.lr_step = self.config['lr_step']
            self.scheduler = LambdaLR(self.optimizer, lambda param: self.step_scheduler_fun(param, self.use_warmup, self.warmup_iter, self.lr_step))
        elif self.lr_strategy == 'linear':
            self.start_epoch = self.config.get('start_epoch', 0)
            self.end_epoch = self.config.get('end_epoch', None)
            self.min_lr = self.config.get('min_lr', 0)
            self.scheduler = LambdaLR(self.optimizer, lambda param: self.linear_scheduler_fun(param, self.use_warmup, self.warmup_iter))
        elif self.lr_strategy is None:
            self.scheduler = LambdaLR(self.optimizer, lambda param: self.freeze_scheduler_fun(param, self.use_warmup, self.warmup_iter))

    def step_scheduler_fun(self, param, use_warmup, warmup_iter, lr_step):
        if isinstance(param, int):
            return 1
        else:
            curr_epoch, curr_iter, max_epoch, max_iter = param
        if use_warmup:
            curr_iter_w_epoch = curr_iter + (curr_epoch-1) * max_iter
            if curr_iter_w_epoch < warmup_iter:
                return curr_iter_w_epoch / warmup_iter
        from bisect import bisect
        idx = bisect(lr_step, curr_epoch)
        return pow(0.1, idx)


    def linear_scheduler_fun(self, param, use_warmup, warmup_iter):
        if isinstance(param, int):
            return 1
        else:
            curr_epoch, curr_iter, max_epoch, max_iter = param
        warmup_epoch = None
        if use_warmup:
            curr_iter_w_epoch = curr_iter + (curr_epoch-1) * max_iter
            if curr_iter_w_epoch < warmup_iter:
                return curr_iter_w_epoch / warmup_iter
            else:
                warmup_epoch = warmup_iter // max_iter + 1
        if warmup_epoch is not None:
            start_epoch = max(self.start_epoch, warmup_epoch)
        else:
            start_epoch = self.start_epoch
        if curr_epoch <= start_epoch:
            return 1
        end_epoch = self.end_epoch if self.end_epoch is not None else max_epoch
        if curr_epoch > end_epoch:
            return self.min_lr / self.lr
        # x 是从 start_epoch 开始之后的 iter_w_epoch
        linear_fn = lambda x: self.lr - x * ((self.lr-self.min_lr) / (end_epoch-start_epoch) / max_iter)
        iter_w_epoch = (curr_epoch-start_epoch-1) * max_iter + curr_iter
        return linear_fn(iter_w_epoch) / self.lr

    def freeze_scheduler_fun(self, param, use_warmup, warmup_iter):
        if isinstance(param, int):
            return 1
        else:
            curr_epoch, curr_iter, max_epoch, max_iter = param
        if use_warmup:
            curr_iter_w_epoch = curr_iter + (curr_epoch-1) * max_iter
            if curr_iter_w_epoch < warmup_iter:
                return curr_iter_w_epoch / warmup_iter
            else:
                return 1
        return 1


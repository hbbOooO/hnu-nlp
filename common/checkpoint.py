import torch
import os

from common.logger import Logger

class CheckPoint:
    
    def __init__(self, ckpt_config, module_config=None):
        self.ckpt_config = ckpt_config
        self.ckpt_dir = ckpt_config['ckpt_dir']
        self.save_by_epoch = ckpt_config.get('save_by_epoch', False)
        self.save_best = ckpt_config.get('save_best', False)
        self.save_last = ckpt_config.get('save_last', False)
        if self.save_by_epoch: self.save_interval = ckpt_config['save_interval']

        self.module_config = module_config
        if module_config:
            self.module_config = module_config
            self.module_dir = module_config['module_dir']
            self.save_module_by_epoch = module_config.get('save_by_epoch', False)
            self.save_module_best = module_config.get('save_best', False)
            self.save_module_last = module_config.get('save_last', False)
            if self.save_module_by_epoch: self.save_module_interval = module_config['save_interval']

    
    def save(self, epoch, model, optimizer, run_config, val_on_val_set, val_metric=None, best_epoch=None, best_val_metric=None):
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'run_config': run_config,
            'val_on_val_set': val_on_val_set,
            'val_metric': val_metric,
            'best_epoch': best_epoch,
            'best_val_metric': best_val_metric,
        }
        ckpt_dir = self.ckpt_dir
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        if self.save_by_epoch and epoch % self.save_interval == 0:
            ckpt_path = ckpt_dir + 'epoch_' + str(epoch) + '.ckpt'
            Logger.get_logger().info('save checkpoint in {}'.format(ckpt_path))
            torch.save(ckpt, ckpt_path)
        if self.save_best and epoch == best_epoch:
            ckpt_path = ckpt_dir + 'epoch_best.ckpt'
            Logger.get_logger().info('save checkpoint in {}'.format(ckpt_path))
            torch.save(ckpt, ckpt_path)
        if  self.save_last:
            ckpt_path = ckpt_dir + 'epoch_last.ckpt'
            Logger.get_logger().info('save checkpoint in {}'.format(ckpt_path))
            torch.save(ckpt, ckpt_path)
    
    def save_module(self, epoch, model, best_epoch=None):
        if self.module_config is None: return
        module_dir = self.module_dir
        if self.save_module_by_epoch and epoch % self.save_module_interval == 0:
            module_dir_by_epoch = module_dir + '/' + str(epoch) + '/'
            if not os.path.exists(module_dir_by_epoch): os.makedirs(module_dir_by_epoch)
            Logger.get_logger().info('save module in {}'.format(module_dir_by_epoch))
            model.save_module(module_dir_by_epoch)
        if self.save_module_best and epoch == best_epoch:
            module_dir_best = module_dir + '/best/'
            if not os.path.exists(module_dir_best): os.makedirs(module_dir_best)
            Logger.get_logger().info('save module in {}'.format(module_dir_best))
            model.save_module(module_dir_best)
        if  self.save_module_last:
            module_dir_last = module_dir + '/last/'
            if not os.path.exists(module_dir_last): os.makedirs(module_dir_last)
            Logger.get_logger().info('save module in {}'.format(module_dir_last))
            model.save_module(module_dir_last)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        return ckpt
import sys
import os
# sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity_error()
import time
import argparse
import json

from common.yml_loader import YmlLoader
from common.trainer import Trainer
from common.logger import Logger
from common.voter import Voter
from common.multi_trainer import MultiTrainer
from common.config_overwrite import config_overwrite


def main():
    args = arg_parser()
    loader = YmlLoader({'yml_path': args.config})
    config = loader()
    overwrite(args, config)
    init_logger(config)
    Logger.get_logger().info('-----------  start hnu-nlp  ------------')
    if args.local_rank != -1:
        multi_trainer = MultiTrainer(config, args)
        multi_trainer()
    elif args.voter:
        voter = Voter(config)
        voter()
    else:
        trainer = Trainer(config, args)
        trainer()
    Logger.get_logger().info('-----------  end hnu-nlp ------------')


def arg_parser():
    parser = argparse.ArgumentParser(description="hnu-nlp")
    parser.add_argument('--config',help='the path of yml file', required=True)
    parser.add_argument('--voter',help='whether to voting inference', action='store_true', default=False)
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--overwrite", help='a string of json type, to overwrite the config of yml file', default=None)
    args = parser.parse_args()
    return args

def overwrite(args, config):
    if args.overwrite is None: return
    overwrite = args.overwrite 
    config_overwrite(overwrite, config)


def init_logger(config):
    log_dir = config['run_param']['log_dir']
    # if the dir is not existing, create it
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_filename = log_dir + time.strftime('%Y-%m-%d %H_%M_%S') + '.log'
    log_level = config['run_param']['log_level']
    Logger.set_config(log_filename, log_level)


if __name__ == '__main__':
    main()
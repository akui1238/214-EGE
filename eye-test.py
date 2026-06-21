import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
# from tensorboardX import SummaryWriter
# #
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

from models.egeunet import EGEUNet

from engine import *
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")




def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    # train_dataset = NPY_datasets(config.data_path, config, train=True)
    # train_loader = DataLoader(train_dataset,
    #                             batch_size=config.batch_size,
    #                             shuffle=True,
    #                             pin_memory=True,
    #                             num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)


    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'egeunet':
        model = EGEUNet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'],
                        c_list=model_cfg['c_list'],
                        bridge=model_cfg['bridge'],
                        gt_ds=model_cfg['gt_ds'],
                        )
        model = nn.parallel.DataParallel(model)
    else: raise Exception('network in not right!')
    model = model.cuda()



    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    # optimizer = get_optimizer(config, model)
    # scheduler = get_scheduler(config, optimizer)


    step = 0
    best_weight_path = '/home/ydm/EGE-UNet-eye/results/egeunet_stare_Monday_10_March_2025_16h_20m_24s/checkpoints/best.pth'
    print('#----------Testing----------#')
    best_weight = torch.load(best_weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(best_weight)
    loss = test_one_epoch(
        val_loader,
        model,
        criterion,
        logger,
        config,
    )
    # os.rename(
    #     os.path.join(checkpoint_dir, 'best.pth'),
    #     os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
    # )


if __name__ == '__main__':
    config = setting_config
    main(config)
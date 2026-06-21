import torch
from torch.utils.data import DataLoader
import timm
import transforms as T
# from datasets.dataset import NPY_datasets
from datasets.datasets1 import DriveDataset, Chasedb1Datasets, STAREDataset, HRFDataset
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
import torch.nn as nn

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


class SegmentationPresetTrain:
    # 用于图像分割任务训练阶段的数据预处理和增强操作。
    def __init__(self, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

# class SegmentationPresetEval:
#     def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#         self.transforms = T.Compose([
#             T.ToTensor(),
#             T.Normalize(mean=mean, std=std),
#         ])
#
#     def __call__(self, img, target):
#         return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    crop_size = 400 #stare
    # crop_size = 720 #chase

    if train:
        return SegmentationPresetTrain(crop_size, mean=mean, std=std)
    else:
        # return SegmentationPresetEval(mean=mean, std=std)
        return SegmentationPresetEval(crop_size, mean=mean, std=std)


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
    # stare
    mean = (0.79472653, 0.42516014, 0.1087752)
    std = (0.17206669, 0.11271852, 0.06384674)
    mean1 = (0.80197152, 0.44979242, 0.13198617)
    std1 = (0.16273552, 0.10759267, 0.04646922)

    # chase
    # mean = (0.44191112, 0.1608891,  0.02802536)
    # std = (0.33401432, 0.13775645, 0.03543717)
    # mean1 = (0.48036714, 0.17151246, 0.02771437)
    # std1 = (0.35466849, 0.14648287, 0.03462432)


    train_dataset = STAREDataset(config.data_path,  train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std)) #+

    # train_dataset = Chasedb1Datasets(config.data_path,  train=True,
    #                              transforms=get_transform(train=True, mean=mean, std=std)) #+

    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                # collate_fn=train_dataset.collate_fn #+
                              )
    val_dataset = STAREDataset(config.data_path,  train=False,
                               transforms=get_transform(train=False, mean=mean1, std=std1))

    # val_dataset = Chasedb1Datasets(config.data_path,  train=False,
    #                            transforms=get_transform(train=False, mean=mean1, std=std1))

    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True,
                                # collate_fn=val_dataset.collate_fn
                            )


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
    best_weight_path = '/home/ydm/EGE-UNet-eye/test-path/best-epoch315-loss1.0103.pth'
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
import argparse
import importlib
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from segmentation3d.dataloader.sampler import EpochConcateSampler
from segmentation3d.loss.focal_loss import FocalLoss
from segmentation3d.loss.multi_dice_loss import MultiDiceLoss
from segmentation3d.utils.file_io import load_config, setup_logger
from segmentation3d.utils.image_tools import save_intermediate_results
from segmentation3d.utils.model_io import load_checkpoint, save_checkpoint

from dataset.dataset import SegmentationDataset
from networks.global_local_net import GlobalLocalNetwork
from networks.module.weight_init import kaiming_weight_init


def train(train_config_file):
    """ Medical image segmentation training engine
    :param train_config_file: the input configuration file
    :return: None
    """
    assert os.path.isfile(train_config_file), 'Config not found: {}'.format(train_config_file)

    # load config file
    train_cfg = load_config(train_config_file)

    # clean the existing folder if training from scratch
    model_folder = os.path.join(train_cfg.general.save_dir, train_cfg.general.model_scale)
    if os.path.isdir(model_folder):
        if train_cfg.general.resume_epoch < 0:
            shutil.rmtree(model_folder)
            os.makedirs(model_folder)
    else:
        os.makedirs(model_folder)

    # copy training and inference config files to the model folder
    shutil.copy(train_config_file, os.path.join(model_folder, 'train_config.py'))
    infer_config_file = os.path.join(os.path.join(os.path.dirname(__file__), 'config', 'infer_config.py'))
    shutil.copy(infer_config_file, os.path.join(train_cfg.general.save_dir, 'infer_config.py'))

    # enable logging
    log_file = os.path.join(model_folder, 'train_log.txt')
    logger = setup_logger(log_file, 'seg3d')

    # control randomness during training
    np.random.seed(train_cfg.general.seed)
    torch.manual_seed(train_cfg.general.seed)
    if train_cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(train_cfg.general.seed)

    # dataset
    dataset = SegmentationDataset(
                mode='train',
                im_list=train_cfg.general.train_im_list,
                num_classes=train_cfg.dataset.num_classes,
                spacing=train_cfg.dataset.spacing,
                crop_size=train_cfg.dataset.crop_size,
                ratio=train_cfg.dataset.down_sample_ratio,
                sampling_method=train_cfg.dataset.sampling_method,
                random_translation=train_cfg.dataset.random_translation,
                random_scale=train_cfg.dataset.random_scale,
                interpolation=train_cfg.dataset.interpolation,
                crop_normalizers=train_cfg.dataset.crop_normalizers)

    sampler = EpochConcateSampler(dataset, train_cfg.train.epochs)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=train_cfg.train.batchsize,
                             num_workers=train_cfg.train.num_threads, pin_memory=True)

    net = GlobalLocalNetwork(dataset.num_modality(), train_cfg.dataset.num_classes)
    net.apply(kaiming_weight_init)
    max_stride = net.max_stride()

    if train_cfg.general.num_gpus > 0:
        net = nn.parallel.DataParallel(net, device_ids=list(range(train_cfg.general.num_gpus)))
        net = net.cuda()

    assert np.all(np.array(train_cfg.dataset.crop_size) % max_stride == 0), 'crop size not divisible by max stride'

    # training optimizer
    opt = optim.Adam(net.parameters(), lr=train_cfg.train.lr, betas=train_cfg.train.betas)

    # load checkpoint if resume epoch > 0
    if train_cfg.general.resume_epoch >= 0:
        last_save_epoch, batch_start = load_checkpoint(train_cfg.general.resume_epoch, net, opt, model_folder)
    else:
        last_save_epoch, batch_start = 0, 0

    if train_cfg.loss.name == 'Focal':
        # reuse focal loss if exists
        loss_func = FocalLoss(class_num=train_cfg.dataset.num_classes, alpha=train_cfg.loss.obj_weight, gamma=train_cfg.loss.focal_gamma,
                              use_gpu=train_cfg.general.num_gpus > 0)
    elif train_cfg.loss.name == 'Dice':
        loss_func = MultiDiceLoss(weights=train_cfg.loss.obj_weight, num_class=train_cfg.dataset.num_classes,
                                  use_gpu=train_cfg.general.num_gpus > 0)
    else:
        raise ValueError('Unknown loss function')

    writer = SummaryWriter(os.path.join(model_folder, 'tensorboard'))

    batch_idx = batch_start
    data_iter = iter(data_loader)

    # loop over batches
    for i in range(len(data_loader)):
        begin_t = time.time()

        crops_o, masks_o, frames_0, crops_g, masks_g, frames_g, crops_l, masks_l, frames_l, coords, filenames = data_iter.next()

        if train_cfg.general.num_gpus > 0:
            crops_g, masks_g, crops_l, masks_l, coords = crops_g.cuda(), masks_g.cuda(), crops_l.cuda(), masks_l.cuda(), coords.cuda()

        # clear previous gradients
        opt.zero_grad()

        # network forward and backward
        outputs_g, outputs_l, outputs_g2l = net(crops_g, crops_l, 3, coords, 4)
        train_loss = loss_func(outputs_g, masks_g)
        train_loss.backward()

        # update weights
        opt.step()

        epoch_idx = batch_idx * train_cfg.train.batchsize // len(dataset)
        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / train_cfg.train.batchsize

        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol'
        msg = msg.format(epoch_idx, batch_idx, train_loss.item(), sample_duration)
        logger.info(msg)


def main():

    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,4,6'

    long_description = "Training engine for 3d medical image segmentation"
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default='/home/ql/projects/Global-Local-Network-3d/config/train_config.py',
                        help='configure file for medical image segmentation training.')
    args = parser.parse_args()
    train(args.input)


if __name__ == '__main__':
    main()

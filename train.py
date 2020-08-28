import argparse
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset.dataset import SegmentationDataset
from loss.focal_loss import FocalLoss
from networks.global_local_net import GlobalLocalNetwork
from networks.module.weight_init import kaiming_weight_init
from utils.file_io import load_config, setup_logger
from utils.helper import Trainer, Evaluator
from utils.image_tools import save_intermediate_results
from utils.model_io import load_checkpoint, save_checkpoint
from utils.metrics import Metrics


def train_one_epoch(model, branch_weight, optimizer, data_loader, down_sample_ratio, loss_func, num_gpus, epoch, logger, writer,
                    print_freq, debug=False, debug_folder=None):
    """ Train one epoch """

    trainer = Trainer(model, optimizer, down_sample_ratio, loss_func, branch_weight, num_gpus > 0)

    avg_loss = 0
    for batch_idx, (crops, masks, frames, names) in enumerate(data_loader):
        begin_t = time.time()
        loss, patches_g, masks_g, out_g, patches_l, masks_l, out_l, out_g2l = trainer.train(crops, masks)
        batch_duration = time.time() - begin_t

        # save training crops for visualization
        if debug:
            assert debug_folder is not None
            batch_size = crops.size(0)

            save_folder = os.path.join(debug_folder, 'images', 'batch_{}'.format(batch_idx))
            save_intermediate_results(list(range(batch_size)), crops, masks, None, frames, names, save_folder)

            save_folder = os.path.join(debug_folder, 'global', 'batch_{}'.format(batch_idx))
            save_intermediate_results(list(range(batch_size)), patches_g, masks_g, out_g, frames, names, save_folder)

            save_folder = os.path.join(debug_folder, 'local', 'batch_{}'.format(batch_idx))
            save_intermediate_results(list(range(batch_size)), patches_l, masks_l, out_g2l, frames, names, save_folder)

        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol'
        msg = msg.format(epoch, batch_idx, loss, batch_duration)
        logger.info(msg)

    writer.add_scalar('Train/Loss', avg_loss / len(data_loader), epoch)


def evaluate_one_epoch(model, data_loader, crop_size, down_sample_ratio, normalizer, metrics, labels):
    """ Evaluate one epoch """

    evaluator = Evaluator(model, metrics, crop_size, down_sample_ratio, normalizer, labels)

    avg_dice = 0
    for batch_idx, (image, mask, name) in enumerate(data_loader):
        dice = evaluator.evaluate(image, mask)
        avg_dice += dice

    return avg_dice / len(data_loader)


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
    train_dataset = SegmentationDataset(
        mode='train',
        im_list=train_cfg.general.train_im_list,
        num_classes=train_cfg.dataset.num_classes,
        spacing=train_cfg.dataset.spacing,
        crop_size=train_cfg.dataset.crop_size,
        sampling_method=train_cfg.dataset.sampling_method,
        random_translation=train_cfg.dataset.random_translation,
        random_scale=train_cfg.dataset.random_scale,
        interpolation=train_cfg.dataset.interpolation,
        crop_normalizers=train_cfg.dataset.crop_normalizers
    )
    train_data_loader = DataLoader(train_dataset, batch_size=train_cfg.train.batchsize,
                                   num_workers=train_cfg.train.num_threads, pin_memory=True, shuffle=True)

    val_dataset = SegmentationDataset(
        mode='val',
        im_list=train_cfg.general.val_im_list,
        num_classes=train_cfg.dataset.num_classes,
        spacing=train_cfg.dataset.spacing,
        crop_size=train_cfg.dataset.crop_size,
        sampling_method=train_cfg.dataset.sampling_method,
        random_translation=train_cfg.dataset.random_translation,
        random_scale=train_cfg.dataset.random_scale,
        interpolation=train_cfg.dataset.interpolation,
        crop_normalizers=train_cfg.dataset.crop_normalizers
    )
    val_data_loader = DataLoader(val_dataset, batch_size=1, num_workers=1, shuffle=False)

    # define network
    net = GlobalLocalNetwork(train_dataset.num_modality(), train_cfg.dataset.num_classes)
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
        last_save_epoch = load_checkpoint(train_cfg.general.resume_epoch, net, opt, model_folder)
    else:
        last_save_epoch = 0

    if train_cfg.loss.name == 'Focal':
        # reuse focal loss if exists
        loss_func = FocalLoss(class_num=train_cfg.dataset.num_classes, alpha=train_cfg.loss.obj_weight,
                              gamma=train_cfg.loss.focal_gamma,use_gpu=train_cfg.general.num_gpus > 0)
    else:
        raise ValueError('Unknown loss function')

    writer = SummaryWriter(os.path.join(model_folder, 'tensorboard'))

    max_avg_dice = 0
    for epoch_idx in range(1, train_cfg.train.epochs + 1):
        train_one_epoch(net, train_cfg.loss.branch_weight, opt, train_data_loader, train_cfg.dataset.down_sample_ratio,
            loss_func, train_cfg.general.num_gpus, epoch_idx+last_save_epoch, logger, writer, train_cfg.train.print_freq,
            train_cfg.debug.save_inputs, os.path.join(model_folder, 'debug'))

        # evaluation
        if epoch_idx % train_cfg.train.save_epochs:
            avg_dice = evaluate_one_epoch(
                net, val_data_loader, train_cfg.dataset.crop_size, train_cfg.dataset.down_sample_ratio,
                train_cfg.dataset.crop_normalizers[0], Metrics(), [idx for idx in range(1, train_cfg.dataset.num_classes)]
            )

            if max_avg_dice < avg_dice:
                max_avg_dice = avg_dice
                save_checkpoint(net, opt, epoch_idx, train_cfg, max_stride, 1)
                msg = 'epoch: {}, best dice ratio: {}'

            else:
                msg = 'epoch: {},  dice ratio: {}'

            msg = msg.format(epoch_idx, avg_dice)
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

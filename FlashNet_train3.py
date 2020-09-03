
from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.getcwd())
# import pdb
# pdb.set_trace()

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import torch.utils.data as data

from FlashNet.facedet.utils.optim import AdamW
from FlashNet.facedet.dataset import LandmarkAnnotationTransform, AnnotationTransform, VOCDetection, \
    detection_collate, preproc_ldmk, preproc, SSDAugmentation
from FlashNet.facedet.losses import MultiBoxLoss
# from losses import FocalLoss
from FlashNet.facedet.utils.anchor.prior_box import PriorBox
from FlashNet.facedet.utils.misc import add_flops_counting_methods, flops_to_string, get_model_parameters_number
from FlashNet.facedet.dataset import data_prefetcher
from FlashNet.facedet.models.flashnet import FlashNet
from FlashNet.facedet.utils.anchor.prior_box import PriorBox
import time
import math
import logging
from datetime import datetime
from mmcv import Config
from dataset import *

os.makedirs("./work_dir/logs/", exist_ok=True)
logging.basicConfig(filename='./work_dir/logs/train_{}.log'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')), level=logging.DEBUG)

torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy('file_system')
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

parser = argparse.ArgumentParser(description='Train anchor-based face detectors')
parser.add_argument('--cfg_file', default='FlashNet/facedet/configs/flashnet_1024_2_anchor.py', type=str,
                    help='model config file')
parser.add_argument('--training_dataset', default='data/WIDER', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=300, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights',
                    help='Location to save checkpoint models')
parser.add_argument('--frozen', default=False, type=bool, help='Froze some layers to finetune model')
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'AdamW'])
parser.add_argument('--gpu_ids', type=str, default='0')
args = parser.parse_args()


class AugmentationCall:
    def __init__(self, func):
        self.func = func

    def __call__(self, image, boxes, labels):
        h, w, _ = image.shape
        boxes[:, 0] *= w
        boxes[:, 1] *= h
        boxes[:, 2] *= w
        boxes[:, 3] *= h
        image, targets = self.func(image, np.hstack((boxes, np.expand_dims(labels, axis=1))))
        image = image.transpose(1, 2, 0)
        image = image[:, :, (2, 1, 0)]
        return image, targets[:, :-1], targets[:, -1]


def train():

    cfg = Config.fromfile('./FlashNet/facedet/configs/flashnet_1024_2_anchor.py')
    rgb_means = (104, 117, 123)
    img_dim = cfg['train_cfg']['input_size']

    dataset = WIDER(dataset='train',
                    image_enhancement_fn=AugmentationCall(preproc(img_dim, rgb_means)),
                    allow_empty_box=False)  # FlashNet not allow training picture without gt box
    dataset_cfg = dataset.cfg


    flash_net = FlashNet(phase='train', cfg=cfg['net_cfg'])
    net = flash_net

    if args.cuda:
        net.cuda()
        cudnn.benchmark = True

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(net.parameters(),
                          lr=args.lr,
                          betas=(0.9, 0.995),
                          eps=1e-9,
                          weight_decay=1e-5,
                          correct_bias=False)
    else:
        raise NotImplementedError('Please use SGD or Adamw as optimizer')

    with torch.no_grad():
        priors = PriorBox(cfg['anchor_cfg']).forward()
        if args.cuda:
            priors = priors.cuda()
    criterion = MultiBoxLoss(2, 0.35, True, 0, True, 3, 0.35, False, cfg['train_cfg']['use_ldmk'])

    net.train()
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)

    for iteration in range(0, 120000):
        if iteration % epoch_size == 0:

            # batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate, drop_last=True))
            if (epoch % 5 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder + 'epoch_' + repr(epoch) + '.pth')
            epoch += 1

        try:
            images, targets = next(batch_iterator)
        except StopIteration as e:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        # load train data
        # images, targets = next(batch_iterator)
        if images is None:
            continue


    torch.save(net.state_dict(), args.save_folder + 'Final_epoch.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 0:
        lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * 5)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
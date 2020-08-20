from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn as nn
import torch.optim as optim
from facedet.utils.optim import AdamW
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import torch.utils.data as data

from dataset import AnnotationTransform, LandmarkAnnotationTransform, FCOSFaceDataset, detection_collate_fcosface, preproc_fcosface
from losses.centerface_losses import *

import time
import math
from facedet.utils.misc import add_flops_counting_methods, flops_to_string, get_model_parameters_number
# from facedet.utils.bbox.fcos_target import FCOSBoxConverter, FCOSBoxTargetConverter
# from tensorboardX import SummaryWriter
# import numpy as np
# writer = SummaryWriter('./log/')

parser = argparse.ArgumentParser(description='CenterFace Training')
parser.add_argument('--cfg_file', default='./configs/centerface_ldmk.py', type=str, help='model config file')
parser.add_argument('--training_dataset', default='/home/gyt/dataset/WIDER_ldmk', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=32, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=300, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--use_tensorboard', dest='use_tensorboard', action='store_true', default=False)
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'AdamW'])
parser.add_argument('--save_folder', default='./weights/xface/',
                    help='Location to save checkpoint models')
args = parser.parse_args()

from mmcv import Config
import logging
cfg = Config.fromfile(args.cfg_file)
logging.basicConfig(filename='./log/train_{}_{}.log'.format(cfg['net_cfg']['net_name'], args.optimizer), level=logging.DEBUG)
args.save_folder = os.path.join(cfg['train_cfg']['save_folder'], args.optimizer)
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
import models

net = models.__dict__[cfg['net_cfg']['net_name']](phase='train', cfg=cfg['net_cfg'])

rgb_means = (104, 117, 123)
img_dim = cfg['train_cfg']['input_size']

batch_size = args.batch_size
weight_decay = args.weight_decay
gamma = args.gamma
momentum = args.momentum

# print("Printing net...")
# print(net)

input_size = (1, 3, img_dim, img_dim)
img = torch.FloatTensor(input_size[0], input_size[1], input_size[2], input_size[3])
net = add_flops_counting_methods(net)
net.start_flops_count()
feat = net(img)
faceboxes_flops = net.compute_average_flops_cost()
print('Net Flops:  {}'.format(flops_to_string(faceboxes_flops)))
print('Net Params: ' + get_model_parameters_number(net))

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

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

box_criterion = LogRegLoss()
ldmk_criterion = RegLoss()
cls_criterion = FocalLoss()

from facedet.utils.bbox.fcosface_target import FCOSFaceTargetGenerator
# target_generator = KPFaceTargetGenerator(stages=3, stride=(8, 16, 32), valid_range=((16, 64), (64, 128), (128, 320)))
target_generator = FCOSFaceTargetGenerator(stages=len(cfg['net_cfg']['strides']), stride=cfg['net_cfg']['strides'], valid_range=((8, 64), (64, 128), (128, 320)))

def train():
    cfg['net_cfg']['use_ldmk']=True
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = FCOSFaceDataset(args.training_dataset,
                                preproc_fcosface(img_dim,
                                               rgb_means,
                                               target_generator=target_generator),
                                LandmarkAnnotationTransform(),
                                aug_type='FaceBoxes',
                                use_ldmk=cfg['net_cfg']['use_ldmk'],
                                ldmk_reg_type=None)


    epoch_size = math.ceil(len(dataset) / args.batch_size)
    max_iter = args.max_epoch * epoch_size

    stepvalues = (200 * epoch_size, 250 * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset,
                                                  batch_size,
                                                  shuffle=True,
                                                  num_workers=args.num_workers,
                                                  collate_fn=detection_collate_fcosface))


            if (epoch % 5 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), os.path.join(args.save_folder, 'FCOSFace_epoch_' + repr(epoch) + '.pth'))
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)
        # load train data
        images, box_targets, box_mask_targets, ldmk_targets, ldmk_mask_targets = next(batch_iterator)

        # print(ldmk_targets[ldmk_mask_targets].min())
        # print(ldmk_targets[ldmk_mask_targets].max())
        # print(ldmk_targets[ldmk_mask_targets])
        # print(box_targets[box_mask_targets].max())
        # print(box_targets[box_mask_targets].min())
        # print(box_targets[box_mask_targets])
        # import pdb
        # pdb.set_trace()
        # continue

        load_t1 = time.time()

        if args.cuda:
            images = Variable(images.cuda())
            box_targets = Variable(box_targets.cuda())
            box_mask_targets = Variable(box_mask_targets.cuda())
            ldmk_targets = Variable(ldmk_targets.cuda())
            ldmk_mask_targets = Variable(ldmk_mask_targets.cuda())
        else:
            images = Variable(images)
            box_targets = Variable(box_targets)
            box_mask_targets = Variable(box_mask_targets)
            ldmk_targets = Variable(ldmk_targets)
            ldmk_mask_targets = Variable(ldmk_mask_targets)

        (box_preds, ldmk_preds, cls_preds) = net(images)

        # backprop
        optimizer.zero_grad()
        loss_cls = cls_criterion(cls_preds, box_mask_targets.float())

        loss_box = box_criterion(pred=box_preds, mask=box_mask_targets, target=box_targets)
        loss_ldmk = ldmk_criterion(pred=ldmk_preds, mask=ldmk_mask_targets, target=ldmk_targets)


        loss = cfg['train_cfg']['box_weight'] * loss_box \
               + cfg['train_cfg']['ldmk_weight'] * loss_ldmk \
               + cfg['train_cfg']['cls_weight'] * loss_cls

        loss.backward()
        optimizer.step()
        load_t2 = time.time()
        if iteration % 50 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size)
                  + '/' + repr(epoch_size) \
                  + ' || BOX: %.3f LDMK: %.3f CLS: %.3f||' % (
                      cfg['train_cfg']['box_weight'] * loss_box.item(),
                      cfg['train_cfg']['ldmk_weight'] * loss_ldmk.item(),
                      cfg['train_cfg']['cls_weight'] * loss_cls.item())
                  + 'Batch time: %.4f sec. ||' % (load_t1 - load_t0)
                  + 'Process time: %.4f sec. ||' % (load_t2 - load_t1)
                  + 'LR: %.8f' % (lr))
            logging.info('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size)
                  + '/' + repr(epoch_size) \
                  + ' || BOX: %.3f LDMK: %.3f CLS: %.3f||' % (
                      cfg['train_cfg']['box_weight'] * loss_box.item(),
                      cfg['train_cfg']['ldmk_weight'] * loss_ldmk.item(),
                      cfg['train_cfg']['cls_weight'] * loss_cls.item())
                  + 'Batch time: %.4f sec. ||' % (load_t1 - load_t0)
                  + 'Process time: %.4f sec. ||' % (load_t2 - load_t1)
                  + 'LR: %.8f' % (lr))

    torch.save(net.state_dict(), os.path.join(args.save_folder, 'Final_FCOSFace.pth'))


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

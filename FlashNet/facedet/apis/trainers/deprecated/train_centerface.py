from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING=1 "] = 1
import torch
import torch.nn as nn
import torch.optim as optim
from facedet.utils.optim import AdamW
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import torch.utils.data as data

from dataset import AnnotationTransform, CenterFaceDataset, detection_collate_centerface, preproc_centerface
from losses.centerface_losses import *

import time
import math
from facedet.utils.misc import add_flops_counting_methods, flops_to_string, get_model_parameters_number
from facedet.utils.bbox.fcos_target import FCOSBoxConverter, FCOSBoxTargetConverter
from tensorboardX import SummaryWriter
import numpy as np
writer = SummaryWriter('./log/')

parser = argparse.ArgumentParser(description='CenterFace Training')
parser.add_argument('--cfg_file', default='./configs/xface_dualpath.py', type=str, help='model config file')
parser.add_argument('--training_dataset', default='/home/gyt/dataset/WIDER_train_5', help='Training dataset directory')
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


cls_criterion = FocalLoss()
wh_criterion = LogRegLoss()
ctr_criterion = RegLoss()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')
    dataset = CenterFaceDataset(args.training_dataset,
                                preproc_centerface(img_dim, rgb_means, use_ldmk=cfg['net_cfg']['use_ldmk']),
                                AnnotationTransform(),
                                aug_type='FaceBoxes')

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
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=args.num_workers,
                                                  collate_fn=detection_collate_centerface))
            if (epoch % 5 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), os.path.join(args.save_folder, 'CenterFace_epoch_' + repr(epoch) + '.pth'))
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, cls_targets, box_targets, ctr_targets, reg_mask_targets = next(batch_iterator)

        # print(images.size())
        # print(cls_targets.size())
        # print(box_targets.size())
        # print(ctr_targets.size())
        # print(reg_mask_targets.size())

        load_t1 = time.time()
        # 写入writer
        if args.batch_size == 1 and args.use_tensorboard:
            img = images.squeeze(0).cpu().numpy().transpose(1, 2, 0) + rgb_means
            img = img.astype(np.uint8).copy()
            img = img.transpose(2, 0, 1).clip(0, 255)
            pos_box_targets = box_targets[(cls_targets > 0).expand_as(box_targets)].view(-1, 4)
            writer.add_image_with_boxes('Image_box', img, pos_box_targets, global_step=iteration, dataformats='CHW')
            writer.add_image('cls0', cls_targets[0][0:256].view(16, 16), global_step=iteration, dataformats='HW')
            writer.add_image('ctr0', ctr_targets[0][0:256].view(16, 16), global_step=iteration, dataformats='HW')
            print('iteration:', iteration)
            continue

        use_anchor_free = ((cls_targets > 0).sum().item() != 0)
        if args.cuda:
            images = Variable(images.cuda())
            box_targets = Variable(box_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())
            ctr_targets = Variable(ctr_targets.cuda())
            reg_mask_targets = Variable(reg_mask_targets.cuda())

        else:
            images = Variable(images)
            box_targets = Variable(box_targets)
            cls_targets = Variable(cls_targets)
            ctr_targets = Variable(ctr_targets)
            reg_mask_targets = Variable(reg_mask_targets)

        (box_preds, cls_preds, ctr_preds) = net(images)

        # backprop
        optimizer.zero_grad()

        # print("box_preds.size()", box_preds.size())
        # print("cls_preds.size()", cls_preds.size())
        # print("ctr_preds.size()", ctr_preds.size())

        # import pdb
        # pdb.set_trace()
        loss_wh = wh_criterion(pred=box_preds, mask=reg_mask_targets, target=box_targets)
        loss_ctr = ctr_criterion(pred=ctr_preds, mask=reg_mask_targets, target=ctr_targets)
        loss_cls = cls_criterion(cls_preds, cls_targets)

        loss = cfg['train_cfg']['wh_weight'] * loss_wh \
               + cfg['train_cfg']['cls_weight'] * loss_cls \
               + cfg['train_cfg']['ctr_weight'] * loss_ctr \

        loss.backward()
        optimizer.step()
        load_t2 = time.time()
        if iteration % 50 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size)
                  + '/' + repr(epoch_size) \
                  + ' || LOC: %.3f CLS: %.3f CTR: %.3f ||' % (
                      cfg['train_cfg']['wh_weight'] * loss_wh.item(),
                      cfg['train_cfg']['cls_weight'] * loss_cls.item(),
                      cfg['train_cfg']['ctr_weight'] * loss_ctr.item())
                  + 'Batch time: %.4f sec. ||' % (load_t1 - load_t0)
                  + 'Process time: %.4f sec. ||' % (load_t2 - load_t1)
                  + 'LR: %.8f' % (lr))
            logging.info('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size)
                  + '/' + repr(epoch_size) \
                  + ' || LOC: %.3f CLS: %.3f CTR: %.3f ||' % (
                      cfg['train_cfg']['wh_weight'] * loss_wh.item(),
                      cfg['train_cfg']['cls_weight'] * loss_cls.item(),
                      cfg['train_cfg']['ctr_weight'] * loss_ctr.item())
                  + 'Batch time: %.4f sec. ||' % (load_t1 - load_t0)
                  + 'Process time: %.4f sec. ||' % (load_t2 - load_t1)
                  + 'LR: %.8f' % (lr))

    torch.save(net.state_dict(), args.save_folder + 'Final_CenterFace.pth')


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

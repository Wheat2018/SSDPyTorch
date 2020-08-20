from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import torch.utils.data as data

from dataset import AnnotationTransform, FCOSFaceDataset, detection_collate_fcos, preproc
from losses import SingleFocalLoss, IOULoss, L1Loss, MultiBoxLoss
from facedet.utils.anchor.prior_box import PriorBox
import time
import math
from facedet.utils.misc import add_flops_counting_methods, flops_to_string, get_model_parameters_number
from facedet.utils.bbox.fcos_target import FCOSBoxConverter, FCOSBoxTargetConverter
from tensorboardX import SummaryWriter
writer = SummaryWriter('./log/')

parser = argparse.ArgumentParser(description='MobileFace Training')
parser.add_argument('--cfg_file', default='./configs/mdface_light.py', type=str, help='model config file')
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
parser.add_argument('--save_folder', default='./weights/fcos/',
                    help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

from mmcv import Config

cfg = Config.fromfile(args.cfg_file)
import models

net = models.__dict__[cfg['net_cfg']['net_name']](phase='train', cfg=cfg['net_cfg'])

rgb_means = (104, 117, 123)
img_dim = cfg['train_cfg']['input_size']

batch_size = args.batch_size
weight_decay = args.weight_decay
gamma = args.gamma
momentum = args.momentum

print("Printing net...")
print(net)

# input_size = (1, 3, img_dim, img_dim)
# img = torch.FloatTensor(input_size[0], input_size[1], input_size[2], input_size[3])
# net = add_flops_counting_methods(net)
# net.start_flops_count()
# feat = net(img)
# faceboxes_flops = net.compute_average_flops_cost()
# print('Net Flops:  {}'.format(flops_to_string(faceboxes_flops)))
# print('Net Params: ' + get_model_parameters_number(net))
# import pdb
# pdb.set_trace()

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

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if cfg['net_cfg']['num_classes'] == 2:
    criterion = MultiBoxLoss(2, 0.35, True, 0, True, 3, 0.35, False)
else:
    loc_criterion = L1Loss()
    iou_criterion = IOULoss()
    cls_criterion = SingleFocalLoss(num_classes=1, overlap_thresh=0.35)
    ctr_criterion = nn.BCEWithLogitsLoss()


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')
    dataset = FCOSFaceDataset(args.training_dataset,
                              preproc(img_dim, rgb_means, is_anchor_free=True),
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
            batch_iterator = iter(data.DataLoader(dataset,
                                                  batch_size,
                                                  shuffle=True,
                                                  num_workers=args.num_workers,
                                                  collate_fn=detection_collate_fcos))
            if (epoch % 1 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder + 'MobileFace_epoch_' + repr(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        import pdb
        pdb.set_trace()
        images, box_targets, cls_targets, ctr_targets, cor_targets = next(batch_iterator)
        load_t1 = time.time()

        if args.cuda:
            images = Variable(images.cuda())
            box_targets = Variable(box_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())
            ctr_targets = Variable(ctr_targets.cuda())
            cor_targets = Variable(cor_targets.cuda())

        else:
            images = Variable(images)
            box_targets = Variable(box_targets)
            cls_targets = Variable(cls_targets)
            ctr_targets = Variable(ctr_targets)
            cor_targets = Variable(cor_targets)

        (box_preds, cls_preds, ctr_preds, detection_dimension) = net(images)

        #         box_preds = FCOSBoxConverter(box_preds, cor_targets)
        # Convert box_targets(x1, y1, x2, y2) to corner(l,t,r,b) format, which then used to compute IoULoss.

        box_targets = FCOSBoxTargetConverter(box_targets, cor_targets)
        #        box_targets = box_targets / img_dim
        #         import pdb
        #         pdb.set_trace()

        # backprop
        optimizer.zero_grad()
        loss_loc = loc_criterion(box_preds, box_targets, cls_targets, weight=None)
        #        loss_iou = iou_criterion(box_preds, box_targets, cls_targets, weight = ctr_targets)
        loss_ctr = ctr_criterion(ctr_preds, ctr_targets)
        loss_cls = cls_criterion(cls_preds, cls_targets)
        #         print(loss_iou.item(), loss_ctr.item(), loss_cls.item())
        #         import pdb
        #         pdb.set_trace()
        loss = cfg['train_cfg']['loc_weight'] * loss_loc \
               + cfg['train_cfg']['cls_weight'] * loss_cls \
               + cfg['train_cfg']['ctr_weight'] * loss_ctr
        loss.backward()
        optimizer.step()
        load_t2 = time.time()
        if iteration % 1 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) \
                  + '/' + repr(epoch_size) \
                  #                   + '|| Totel iter ' + repr(iteration) \
                  + ' || LOC: %.3f CLS: %.3f CTR: %.3f||' % (
                      cfg['train_cfg']['loc_weight'] * loss_loc.item(), \
                      cfg['train_cfg']['cls_weight'] * loss_cls.item(),
                      cfg['train_cfg']['ctr_weight'] * loss_ctr.item()) \
                  + 'Batch time: %.4f sec. ||' % (load_t1 - load_t0) \
                  + 'Process time: %.4f sec. ||' % (load_t2 - load_t1) \
                  + 'LR: %.8f' % (lr))

    torch.save(net.state_dict(), args.save_folder + 'Final_MobileFace.pth')


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

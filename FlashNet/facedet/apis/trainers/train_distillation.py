from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import torch.optim as optim
from facedet.utils.optim import AdamW
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from dataset import AnnotationTransform, VOCDetection, detection_collate, preproc, SSDAugmentation
from losses import SingleFocalLoss, IOULoss, MultiBoxLoss, DistillationLoss
from facedet.utils.anchor.prior_box import PriorBox
import time
import math
import models
import logging
from facedet.utils.misc import add_flops_counting_methods, flops_to_string, get_model_parameters_number
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='FaceBoxes Training')
parser.add_argument('--tch_cfg_file', default='./configs/mdface_2x_1024.py', type=str,
                    help='model config file')
parser.add_argument('--stu_cfg_file', default='./configs/mdface_light_1024.py', type=str,
                    help='model config file')
parser.add_argument('--training_dataset', default='/mnt/lustre/geyongtao/dataset/WIDER_train_5',
                    help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=300, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/mobileface_distillation_2x_1x/',
                    help='Location to save checkpoint models')
parser.add_argument('--tch_weight', default='./weights/MDFace_2x_1024/epoch_299.pth',
                    help='Location to load teacher weight')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

from mmcv import Config

cfg = Config.fromfile(args.stu_cfg_file)
logging.basicConfig(filename='./log/train_distillation_adamw_{}.log'.format(cfg['net_cfg']['net_name']), level=logging.DEBUG)
args.save_folder = cfg['train_cfg']['save_folder']
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
net = models.__dict__[cfg['net_cfg']['net_name']](phase='train', cfg=cfg['net_cfg'])

rgb_means = (104, 117, 123)
img_dim = cfg['train_cfg']['input_size']
batch_size = args.batch_size
weight_decay = args.weight_decay
gamma = args.gamma
momentum = args.momentum

cfg_tch = Config.fromfile(args.tch_cfg_file)
net_tch = models.__dict__[cfg_tch['net_cfg']['net_name']](phase='train', cfg=cfg_tch['net_cfg'])

# print("Printing net...")
# print(net)

# input_size = (1, 3, img_dim, img_dim)
# img = torch.FloatTensor(input_size[0], input_size[1], input_size[2], input_size[3])

# net_tch = add_flops_counting_methods(net_tch)
# net_tch.start_flops_count()
# net = add_flops_counting_methods(net)
# net.start_flops_count()

# _ = net_tch(img)
# tch_flops = net_tch.compute_average_flops_cost()
# print('MobileFace teacher Flops:  {}'.format(flops_to_string(tch_flops)))
# print('MobileFace teacher Params: ' + get_model_parameters_number(net_tch))

# stu_flops = net.compute_average_flops_cost()
# print('MobileFace student Flops:  {}'.format(flops_to_string(stu_flops)))
# print('MobileFace student Params: ' + get_model_parameters_number(net))

print('Loading teacher network...')
state_dict = torch.load(args.tch_weight)
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
net_tch.load_state_dict(new_state_dict)

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
    net_tch = torch.nn.DataParallel(net_tch, device_ids=list(range(args.ngpu)))
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.cuda:
    net_tch.cuda()
    net.cuda()
    cudnn.benchmark = True

# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = AdamW(net.parameters(),
                  lr=args.lr,
                  betas=(0.9, 0.995),
                  eps=1e-9,
                  weight_decay=1e-5,
                  correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False


# criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
criterion = DistillationLoss(cfg['net_cfg']['num_classes'], 0.35, True, 0, True, 3, 0.35, False)

priorbox = PriorBox(cfg['anchor_cfg'])
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


def train():
    net_tch.eval()
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = VOCDetection(args.training_dataset, preproc(img_dim, rgb_means), AnnotationTransform())
    #     dataset = VOCDetection(args.training_dataset, SSDAugmentation(img_dim, rgb_means), AnnotationTransform())

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
                                                  collate_fn=detection_collate))
            if (epoch % 3 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), os.path.join(args.save_folder, 'epoch_' + repr(epoch) + '.pth'))
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]

        # forward
        out_tch = net_tch(images)
        out = net(images)

        # backprop
        optimizer.zero_grad()
        #         import pdb
        #         pdb.set_trace()
        loss_l, loss_c, loss_c_distillation = criterion(out, out_tch, priors, targets)
        loss = cfg['train_cfg']['loc_weight'] * loss_l \
               + cfg['train_cfg']['cls_weight'] * loss_c \
               + cfg['train_cfg']['distillation_weight'] * loss_c_distillation

        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        if iteration % 50 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) +
                  '|| Totel iter ' + repr(iteration) + ' || L: %.4f C: %.4f D: %.4f||' \
                  % (cfg['train_cfg']['loc_weight'] * loss_l.item(), \
                     cfg['train_cfg']['cls_weight'] * loss_c.item(), \
                     cfg['train_cfg']['distillation_weight'] * loss_c_distillation.item()) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
            logging.info(('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) +
                  '|| Totel iter ' + repr(iteration) + ' || L: %.4f C: %.4f D: %.4f||' \
                  % (cfg['train_cfg']['loc_weight'] * loss_l.item(), \
                     cfg['train_cfg']['cls_weight'] * loss_c.item(), \
                     cfg['train_cfg']['distillation_weight'] * loss_c_distillation.item()) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr)))

    torch.save(net.state_dict(), os.path.join(args.save_folder, 'final_epoch.pth'))


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

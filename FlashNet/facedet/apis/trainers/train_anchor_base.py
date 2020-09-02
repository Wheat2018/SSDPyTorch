
from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.getcwd())
# import pdb
# pdb.set_trace()

import torch
import torch.optim as optim
from FlashNet.facedet.utils.optim import AdamW
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import torch.utils.data as data

from FlashNet.facedet.dataset import LandmarkAnnotationTransform, AnnotationTransform, VOCDetection, \
    detection_collate, preproc_ldmk, preproc, SSDAugmentation
from FlashNet.facedet.losses import MultiBoxLoss
# from losses import FocalLoss
from FlashNet.facedet.utils.anchor.prior_box import PriorBox
import time
import math
from FlashNet.facedet.utils.misc import add_flops_counting_methods, flops_to_string, get_model_parameters_number
from FlashNet.facedet.dataset import data_prefetcher
import logging
from datetime import datetime

os.makedirs("./work_dir/logs/", exist_ok=True)
logging.basicConfig(filename='./work_dir/logs/train_{}.log'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')), level=logging.DEBUG)

torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy('file_system')
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

parser = argparse.ArgumentParser(description='Train anchor-based face detectors')
parser.add_argument('--cfg_file', default='../../configs/flashnet_1024_2_anchor.py', type=str,
                    help='model config file')
parser.add_argument('--training_dataset', default='../../../../data/WIDER', help='Training dataset directory')
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
parser.add_argument('--save_folder', default='../../../../weights',
                    help='Location to save checkpoint models')
parser.add_argument('--frozen', default=False, type=bool, help='Froze some layers to finetune model')
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'AdamW'])
parser.add_argument('--gpu_ids', type=str, default='0')
args = parser.parse_args()


def train():
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    from mmcv import Config

    cfg = Config.fromfile(args.cfg_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    args.save_folder = os.path.join(cfg['train_cfg']['save_folder'], args.optimizer)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    import FlashNet.facedet.models as models

    net = models.__dict__[cfg['net_cfg']['net_name']](phase='train', cfg=cfg['net_cfg'])

    rgb_means = (104, 117, 123)
    img_dim = cfg['train_cfg']['input_size']

    batch_size = args.batch_size
    weight_decay = args.weight_decay
    gamma = args.gamma
    momentum = args.momentum

    print("Printing net...")
    # print(net)
    # img_dim = 1024
    input_size = (1, 3, img_dim, img_dim)

    img = torch.FloatTensor(input_size[0], input_size[1], input_size[2], input_size[3])
    net = add_flops_counting_methods(net)
    net.start_flops_count()
    feat = net(img)
    flops = net.compute_average_flops_cost()
    print('Net Flops:  {}'.format(flops_to_string(flops)))
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
        net.load_state_dict(new_state_dict, strict=False)

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

    if cfg['net_cfg']['num_classes'] == 2:
        criterion = MultiBoxLoss(2, 0.35, True, 0, True, 3, 0.35, False, cfg['train_cfg']['use_ldmk'])
    else:
        criterion = FocalLoss(num_classes=1, overlap_thresh=0.35)

    priorbox = PriorBox(cfg['anchor_cfg'])

    with torch.no_grad():
        priors = priorbox.forward()
        if args.cuda:
            priors = priors.cuda()

    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')
    anchors = cfg['anchor_cfg']['anchors']

    if cfg['train_cfg']['use_ldmk']:
        dataset = VOCDetection(args.training_dataset, preproc_ldmk(img_dim, rgb_means),
                               LandmarkAnnotationTransform())
    else:
        dataset = VOCDetection(args.training_dataset, preproc(img_dim, rgb_means), AnnotationTransform())

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
            train_loader = data.DataLoader(dataset, batch_size, shuffle=True, \
                                           num_workers=args.num_workers, collate_fn=detection_collate, drop_last=True)
            prefetcher = data_prefetcher(train_loader)

            # batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate, drop_last=True))
            if (epoch % 5 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder + 'epoch_' + repr(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        # images, targets = next(batch_iterator)
        images, targets = prefetcher.next()
        if images is None:
            continue

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]

        # forward
        # img = images.squeeze(0).cpu().numpy().transpose(1, 2, 0) + rgb_means
        # import numpy as np
        # img = img.astype(np.uint8).copy()
        #
        # import cv2
        # boxes = targets[0][:, 0:4].reshape(-1,4).cpu().numpy()
        #
        # boxes = boxes * 640
        # for box in boxes:
        #     xmin, ymin, xmax, ymax = np.ceil(box)
        #     cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 255), 2)
        #
        # if not os.path.exists('show_train_img_with_anno'):
        #     os.makedirs('show_train_img_with_anno')
        #
        # # print(os.path.join('./show_train_img_with_anno/', str(iteration) + '.jpg'))
        # cv2.imwrite(os.path.join('./show_train_img_with_anno/', str(iteration)+'no_rotate'+'.jpg'), img)
        # print(iteration)
        # continue

        out = net(images)

        # backprop
        optimizer.zero_grad()
        if cfg['train_cfg']['use_ldmk']:
            loss_landmark, loss_l, loss_c = criterion(out, priors, targets)
            loss = cfg['train_cfg']['landmark_weight'] * loss_landmark + \
                   cfg['train_cfg']['loc_weight'] * loss_l + \
                   cfg['train_cfg']['cls_weight'] * loss_c
        else:
            loss_l, loss_c = criterion(out, priors, targets)
            loss = cfg['train_cfg']['loc_weight'] * loss_l + cfg['train_cfg']['cls_weight'] * loss_c

        loss.backward()
        optimizer.step()


        # from facedet.utils.misc.vis_cnn import make_dot
        # g = make_dot(loss_l)
        # g.format = 'pdf'
        # g.render('loss_loc')
        # g = make_dot(loss_c)
        # g.format = 'pdf'
        # g.render('loss_cls')
        # g = make_dot(loss_landmark)
        # g.format = 'pdf'
        # g.render('loss_landmark')
        #
        # import pdb
        # pdb.set_trace()

        load_t1 = time.time()
        if iteration % 10 == 0:
            if cfg['train_cfg']['use_ldmk']:
                logging.info('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size)
                      + '/' + repr(epoch_size)
                      + ' ||Landmark: %.4f L: %.4f C: %.4f||' % (cfg['train_cfg']['landmark_weight'] * loss_landmark.item(),
                                                                  cfg['train_cfg']['loc_weight'] * loss_l.item(),
                                                                cfg['train_cfg']['cls_weight'] * loss_c.item())
                      + 'Batch time: %.4f sec. ||' % (load_t1 - load_t0) \
                      + 'LR: %.8f' % (lr))

                print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) \
                      + '/' + repr(epoch_size) \
                      + ' ||Landmark: %.4f L: %.4f C: %.4f||' % (cfg['train_cfg']['landmark_weight'] * loss_landmark.item(),
                                                                  cfg['train_cfg']['loc_weight'] * loss_l.item(), \
                                                                cfg['train_cfg']['cls_weight'] * loss_c.item()) \
                      + 'Batch time: %.4f sec. ||' % (load_t1 - load_t0) \
                      + 'LR: %.8f' % (lr))
            else:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) \
                      + '/' + repr(epoch_size) \
                      + '|| Totel iter ' + repr(iteration) \
                      + ' || L: %.4f C: %.4f||' % (cfg['train_cfg']['loc_weight'] * loss_l.item(), \
                                                   cfg['train_cfg']['cls_weight'] * loss_c.item()) \
                      + 'Batch time: %.4f sec. ||' % (load_t1 - load_t0) \
                      + 'LR: %.8f' % (lr))

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
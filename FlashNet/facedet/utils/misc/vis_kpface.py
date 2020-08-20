from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import torch.utils.data as data

from dataset import AnnotationTransform, LandmarkAnnotationTransform, KPFaceDataset, detection_collate_kpface, preproc_kpface_vis
from losses.centerface_losses import *

import time
import math
from facedet.utils.misc import add_flops_counting_methods, flops_to_string, get_model_parameters_number
from facedet.utils.bbox.fcos_target import FCOSBoxConverter, FCOSBoxTargetConverter
from tensorboardX import SummaryWriter
import numpy as np
import cv2

writer = SummaryWriter('./log/vis_kpface')

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

from facedet.utils.bbox.kpface_target import KPFaceTargetGenerator
# target_generator = KPFaceTargetGenerator(stages=3, stride=(8, 16, 32), valid_range=((16, 64), (64, 128), (128, 320)))
target_generator = KPFaceTargetGenerator(stages=len(cfg['net_cfg']['strides']), stride=cfg['net_cfg']['strides'], valid_range=((16, 64), (64, 128), (128, 320)))


def vis():
    cfg['net_cfg']['use_ldmk']=True
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = KPFaceDataset(args.training_dataset,
                                preproc_kpface_vis(img_dim,
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
                                                  collate_fn=detection_collate_kpface))


            if (epoch % 5 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), os.path.join(args.save_folder, 'KPFace_epoch_' + repr(epoch) + '.pth'))
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        # load vis data
        images, box_targets, box_mask_targets, ldmk_targets, hard_face_targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            box_targets = Variable(box_targets.cuda())
            box_mask_targets = Variable(box_mask_targets.cuda())
            ldmk_targets = Variable(ldmk_targets.cuda())
            hard_face_targets = Variable(hard_face_targets.cuda())
        else:
            images = Variable(images)
            box_targets = Variable(box_targets)
            box_mask_targets = Variable(box_mask_targets)
            ldmk_targets = Variable(ldmk_targets)
            hard_face_targets = Variable(hard_face_targets)

        (box_preds, ldmk_preds, hard_face_preds) = net(images)
        # 写入writer
        # import pdb
        # pdb.set_trace()
        if args.batch_size == 1 and args.use_tensorboard:
            img = images.squeeze(0).cpu().numpy().transpose(1, 2, 0) + rgb_means
            img = img.astype(np.uint8).copy()
            #bgr->rgb



            if cfg['net_cfg']['strides'][0] == 8:
                writer.add_image('box_mask_tgt_s8', box_mask_targets[0][0:6400].view(80, 80).float(), global_step=iteration, dataformats='HW')

                ldmk_gt = torch.cat([ldmk_targets[0][0:6400][:, 0].view(160, 160),
                                       ldmk_targets[0][0:6400][:, 1].view(160, 160),
                                       ldmk_targets[0][0:6400][:, 2].view(160, 160),
                                       ldmk_targets[0][0:6400][:, 3].view(160, 160),
                                       ldmk_targets[0][0:6400][:, 4].view(160, 160)], dim=1).float()

                ldmk_pred = torch.cat([ldmk_preds[0][0:6400][:, 0].view(160, 160),
                                       ldmk_preds[0][0:6400][:, 1].view(160, 160),
                                       ldmk_preds[0][0:6400][:, 2].view(160, 160),
                                       ldmk_preds[0][0:6400][:, 3].view(160, 160),
                                       ldmk_preds[0][0:6400][:, 4].view(160, 160)], dim=1).float()
                ldmk_pred[ldmk_pred < 0.25] = 0
                ldmk = torch.cat([ldmk_gt, ldmk_pred], dim=0)

                writer.add_image('ldmk_tgt_s4', ldmk, global_step=iteration,
                                 dataformats='HW')
            if cfg['net_cfg']['strides'][0] == 4:
                mask_s4 = box_mask_targets[0][0:25600].view(160, 160).cpu().numpy()
                mask_s8 = box_mask_targets[0][25600:32000].view(80, 80).cpu().numpy()
                mask_s16 = box_mask_targets[0][32000:33600].view(40, 40).cpu().numpy()
                # import pdb
                # pdb.set_trace()
                mask_s4 = cv2.resize(mask_s4, (640, 640), interpolation=cv2.INTER_CUBIC)
                mask_s8 = cv2.resize(mask_s8, (640, 640), interpolation=cv2.INTER_CUBIC)
                mask_s16 = cv2.resize(mask_s16, (640, 640), interpolation=cv2.INTER_CUBIC)
                img_s4 = cv2.bitwise_and(img, img, mask=mask_s4)[...,::-1]
                img_s8 = cv2.bitwise_and(img, img, mask=mask_s8)[...,::-1]
                img_s16 = cv2.bitwise_and(img, img, mask=mask_s16)[...,::-1]
                img = img[...,::-1]

                # img_s4 = img * mask_s4
                # img_s8 = img * mask_s8
                # img_s16 = img * mask_s16
                img_concat = np.concatenate((img, img_s4, img_s8, img_s16), axis=1)
                writer.add_image('Image', img_concat, global_step=iteration, dataformats='HWC')
                # import pdb
                # pdb.set_trace()
                #
                # writer.add_image('box_mask_tgt_s4', box_mask_targets[0][0:25600].view(160, 160).float(),
                #                  global_step=iteration, dataformats='HW')
                # writer.add_image('box_mask_tgt_s8', box_mask_targets[0][25600:32000].view(80, 80).float(),
                #                  global_step=iteration, dataformats='HW')
                # writer.add_image('box_mask_tgt_s16', box_mask_targets[0][32000:33600].view(40, 40).float(),
                #                  global_step=iteration, dataformats='HW')

                ldmk_gt = torch.cat([ldmk_targets[0][0:25600][:, 0].view(160, 160),
                                     ldmk_targets[0][0:25600][:, 1].view(160, 160),
                                     ldmk_targets[0][0:25600][:, 2].view(160, 160),
                                     ldmk_targets[0][0:25600][:, 3].view(160, 160),
                                     ldmk_targets[0][0:25600][:, 4].view(160, 160)], dim=1).float()

                ldmk_pred = torch.cat([ldmk_preds[0][0:25600][:, 0].view(160, 160),
                                       ldmk_preds[0][0:25600][:, 1].view(160, 160),
                                       ldmk_preds[0][0:25600][:, 2].view(160, 160),
                                       ldmk_preds[0][0:25600][:, 3].view(160, 160),
                                       ldmk_preds[0][0:25600][:, 4].view(160, 160)], dim=1).float()


                ldmk_pred[ldmk_pred < 0.1] = 0
                ldmk_s4 = torch.cat([ldmk_gt, ldmk_pred], dim=0)

                writer.add_image('ldmk_tgt_s4', ldmk_s4, global_step=iteration,
                                 dataformats='HW')

                ldmk_gt = torch.cat([ldmk_targets[0][25600:32000][:, 0].view(80, 80),
                                     ldmk_targets[0][25600:32000][:, 1].view(80, 80),
                                     ldmk_targets[0][25600:32000][:, 2].view(80, 80),
                                     ldmk_targets[0][25600:32000][:, 3].view(80, 80),
                                     ldmk_targets[0][25600:32000][:, 4].view(80, 80)], dim=1).float()

                ldmk_pred = torch.cat([ldmk_preds[0][25600:32000][:, 0].view(80, 80),
                                       ldmk_preds[0][25600:32000][:, 1].view(80, 80),
                                       ldmk_preds[0][25600:32000][:, 2].view(80, 80),
                                       ldmk_preds[0][25600:32000][:, 3].view(80, 80),
                                       ldmk_preds[0][25600:32000][:, 4].view(80, 80)], dim=1).float()


                ldmk_pred[ldmk_pred < 0.1] = 0
                ldmk_s8 = torch.cat([ldmk_gt, ldmk_pred], dim=0)

                writer.add_image('ldmk_tgt_s8', ldmk_s8, global_step=iteration,
                                 dataformats='HW')


                ldmk_gt = torch.cat([ldmk_targets[0][32000:33600][:, 0].view(40, 40),
                                     ldmk_targets[0][32000:33600][:, 1].view(40, 40),
                                     ldmk_targets[0][32000:33600][:, 2].view(40, 40),
                                     ldmk_targets[0][32000:33600][:, 3].view(40, 40),
                                     ldmk_targets[0][32000:33600][:, 4].view(40, 40)], dim=1).float()

                ldmk_pred = torch.cat([ldmk_preds[0][32000:33600][:, 0].view(40, 40),
                                       ldmk_preds[0][32000:33600][:, 1].view(40, 40),
                                       ldmk_preds[0][32000:33600][:, 2].view(40, 40),
                                       ldmk_preds[0][32000:33600][:, 3].view(40, 40),
                                       ldmk_preds[0][32000:33600][:, 4].view(40, 40)], dim=1).float()


                ldmk_pred[ldmk_pred < 0.1] = 0
                ldmk_s16 = torch.cat([ldmk_gt, ldmk_pred], dim=0)

                writer.add_image('ldmk_tgt_s16', ldmk_s16, global_step=iteration,
                                 dataformats='HW')


                hard_face_gt = hard_face_targets[0][0:25600][:, 0].view(160, 160)
                writer.add_image('hard_face_tgt_s4', hard_face_gt, global_step=iteration,
                                 dataformats='HW')

                hard_face_gt = hard_face_targets[0][25600:32000][:, 0].view(80, 80)
                writer.add_image('hard_face_tgt_s8', hard_face_gt, global_step=iteration,
                                 dataformats='HW')

                hard_face_gt = hard_face_targets[0][32000:33600][:, 0].view(40, 40)
                writer.add_image('hard_face_tgt_s16', hard_face_gt, global_step=iteration,
                                 dataformats='HW')

            print('iteration:', iteration)


if __name__ == '__main__':
    vis()

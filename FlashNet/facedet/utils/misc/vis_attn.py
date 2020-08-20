'''
srun --partition=Test --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=test --kill-on-bad-exit=1 python vis_attn.py --cfg_file ./configs/mdface_light_1024.py --resume_net ./weights/MDFace_light_1024/epoch_200.pth --ngpu 1 --resume_epoch 200
'''
from __future__ import print_function
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim as optim
from facedet.utils.optim import AdamW
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import torch.utils.data as data

from dataset import LandmarkAnnotationTransform, AnnotationTransform, VOCDetection, \
    detection_collate, landmark_preproc, preproc, SSDAugmentation
from losses import MultiBoxLoss
# from losses import FocalLoss
from facedet.utils.anchor.prior_box import PriorBox
import time
import math
from facedet.utils.misc import add_flops_counting_methods, flops_to_string, get_model_parameters_number,get_model_complexity_info
from dataset import data_prefetcher
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import requests
import numpy as np
from io import BytesIO
import torch
from torch import nn
from torchvision.models import resnet34
from torchvision.models.resnet import ResNet, BasicBlock
import torchvision.transforms as T
import torch.nn.functional as F
import cv2

# logging.basicConfig(filename='./log/train_{}.log'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')), level=logging.DEBUG)

torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy('file_system')
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

parser = argparse.ArgumentParser(description='MobileFace Training')
parser.add_argument('--cfg_file', default='./configs/mdface_light_1024.py', type=str,
                    help='model config file')
parser.add_argument('--training_dataset', default='/mnt/lustre/geyongtao/dataset/WIDER_train_5', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=24, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=300, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/SwitchNet/',
                    help='Location to save checkpoint models')
parser.add_argument('--frozen', default=False, type=bool, help='Froze some layers to finetune model')
args = parser.parse_args()

from mmcv import Config

cfg = Config.fromfile(args.cfg_file)
cfg['net_cfg']['out_featmaps']=True
logging.basicConfig(filename='./log/train_{}.log'.format(cfg['net_cfg']['net_name']), level=logging.DEBUG)
args.save_folder = cfg['train_cfg']['save_folder']
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
# input_size = (1, 3, img_dim, img_dim)
# flops, params = get_model_complexity_info(net, (3, img_dim, img_dim))
# split_line = '=' * 30
# print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
#     split_line, input_size, flops, params))
#
# img = torch.FloatTensor(input_size[0], input_size[1], input_size[2], input_size[3])
# net = add_flops_counting_methods(net)
# net.start_flops_count()
# feat = net(img)
# faceboxes_flops = net.compute_average_flops_cost()
# print('Net Flops:  {}'.format(flops_to_string(faceboxes_flops)))
# print('Net Params: ' + get_model_parameters_number(net))


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



def transform_image(image):                # 定义转化函数，将PIL格式的图像转化为格式维度的numpy格式数组
    mean = [123., 117., 104.]  # 在ImageNet上训练数据集的mean和std
    mean = [104., 117., 123.]  # 在ImageNet上训练数据集的mean和std
    # std = [58.395, 57.12, 57.375]
    image = image - np.array(mean)
    # image /= np.array(std)
    # image = np.array(image).transpose((2, 0, 1))
    # image = image[np.newaxis, :].astype('float32')
    return image


def vis():
    net.eval()

    # tr_center_crop = T.Compose([
    #     # T.ToPILImage(),
    #     # T.Resize(256),
    #     T.ToTensor(),
    #     # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    #
    # img = Image.open('./speed_benchmark/26_Soldier_Drilling_Soldiers_Drilling_26_178.jpg').resize((640, 480))  # 这里我们将图像resize为特定大小
    # # img = Image.open('./speed_benchmark/4_Dancing_Dancing_4_690.jpg').resize((320, 256)) # 这里我们将图像resize为特定大小
    # img = transform_image(img)
    # with torch.no_grad():
    #     x = tr_center_crop(img).unsqueeze(0).float()
    #     if args.cuda:
    #         x = x.cuda()
    #     _, feats = net(x)

    image_path='./speed_benchmark/0_Parade_Parade_0_470.jpg'
    # image_path = './speed_benchmark/19_Couple_Couple_19_822.jpg'
    img = np.float32(cv2.imread(image_path, cv2.IMREAD_COLOR))
    # import pdb
    # pdb.set_trace()
    width = 640
    height = 640
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    original_img = img.copy()
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    x = torch.from_numpy(img).unsqueeze(0)
    with torch.no_grad():
        if args.cuda:
            x = x.cuda()
        _, feats = net(x)

    #[1, 60, 80] [1, 30, 40] [1, 15, 20]
    if args.cuda:
        feats = [feat.pow(2).mean(1) for feat in feats]
    else:
        feats = [feat.pow(2).mean(1).cpu() for feat in feats]



    # for i, feat in enumerate(feats):
    #     # plt.subplot(1, 3, i)
    #     # plt.imshow(feat[0], interpolation='bicubic', cmap="gray")
    #     # plt.imshow(feat[0], interpolation='bicubic')
    #     # import matplotlib.image as Image
    #     # Image.imread(image_path).convert("L")
    #     img = Image.open(image_path).convert("L").resize((feat[0].shape[0], feat[0].shape[1]))
    #     import pdb
    #     pdb.set_trace()
    #     alpha = 0.0001
    #     plt.imshow(np.array(img),cmap="gray")
    #     # plt.imshow(alpha * np.array(feat[0]) + (1-alpha)*np.array(img))
    #     # plt.title(cfg['net_cfg']['net_name']+'_' + f'feat{i}'+f'epoch{args.resume_epoch}')
    #     plt.axis('off')
    #     plt.gcf().savefig(cfg['net_cfg']['net_name'] + '_' + f'feat{i}' +'_' + f'epoch{args.resume_epoch}' + '_mixup.jpg', bbox_inches='tight')
    #     plt.clf()



    for i, feat in enumerate(feats):
        # import pdb
        # pdb.set_trace()
        feat = feat / feat.max()
        feat = feat.permute(1, 2, 0).numpy()
        # feat /= np.max(feat)
        feat = cv2.resize(feat, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*feat), cv2.COLORMAP_JET)
        threshold = 0.1
        heatmap[np.where(feat < threshold)] = 0
        img = heatmap*0.5 + original_img
        output_path = cfg['net_cfg']['net_name'] + '_' + f'feat{i}' +'_' + f'epoch{args.resume_epoch}' + f'_threshold{threshold}' + '_mixup.jpg'
        cv2.imwrite(output_path, img)
        # plt.show()




if __name__ == '__main__':
    vis()
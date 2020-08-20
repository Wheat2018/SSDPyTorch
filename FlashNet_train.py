"""
    By.Wheat
    2020.07.29
"""
from dataset import *
from utils import *
from ssd_face import *
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse
import base_nets

from mmcv import Config
from FlashNet.facedet.utils.optim.optimization import AdamW
from FlashNet.facedet.losses.multibox_loss import MultiBoxLoss
from FlashNet.facedet.utils.anchor.prior_box import PriorBox
from FlashNet.facedet.dataset.transform.data_augment import preproc
from FlashNet.facedet.models.flashnet import FlashNet

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='WIDER', choices=['FDDB', 'WIDER'],
                    type=str, help='FDDB or WIDER')
# parser.add_argument('--dataset_root', default=FDDB_ROOT,
#                     help='Dataset root directory path')
parser.add_argument('--ssd_weight', default='SSDFace300_VGG_FDDB_fromtrain1.pth',
                    help='ssd model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=24, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--optimizer', type=str, default='AdamW',
                    choices=['SGD', 'AdamW'])
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

viz = None
if args.visdom:
    import visdom

    viz = visdom.Visdom()


class AugmentationCall:
    def __init__(self, func):
        self.func = func

    def __call__(self, image, boxes, labels):
        image, targets = self.func(image, np.hstack((boxes, np.expand_dims(labels, axis=1))))
        image = image.transpose(1, 2, 0)
        image = image[:, :, (2, 1, 0)]
        return image, targets[:, :-1], targets[:, -1]


def train():
    import FlashNet.facedet.configs.flashnet_1024_2_anchor as fl_cfg
    cfg = {
        'net_cfg': fl_cfg.net_cfg,
        'anchor_cfg': fl_cfg.anchor_cfg,
        'train_cfg': fl_cfg.train_cfg,
        'test_cfg': fl_cfg.test_cfg
    }
    # cfg = Config.fromfile('./FlashNet/facedet/configs/flashnet_1024_2_anchor.py')
    rgb_means = (104, 117, 123)
    img_dim = cfg['train_cfg']['input_size']

    if args.dataset == 'FDDB':
        dataset = FDDB(dataset='train',
                       image_enhancement_fn=AugmentationCall(preproc(img_dim, rgb_means)))
        dataset_cfg = dataset.cfg
    elif args.dataset == 'WIDER':
        dataset = WIDER(dataset='train',
                        image_enhancement_fn=AugmentationCall(preproc(img_dim, rgb_means)))
        dataset_cfg = dataset.cfg

    else:
        dataset = None
        dataset_cfg = None

    dataset.pull_item(0)
    flash_net = FlashNet(phase='train', cfg=cfg['net_cfg'])
    net = flash_net

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        base_nets.load_weights(net, args.resume)
    else:
        base_nets.load_weights(net, path.join(args.save_folder, args.ssd_weight))

    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net = net.cuda()

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
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

    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    iter_plot = None
    epoch_plot = None
    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)
    print('len:', len(dataset))
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, dataset_cfg['max_iter']):

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration as e:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
            epoch += 1
            print('epoch:%d\t|| loc/conf/all: %.4f / %.4f / %.4f' %
                  (epoch, loc_loss, conf_loss, loc_loss+conf_loss))
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            if args.visdom:
                try:
                    update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                                    'append', epoch_size)
                except Exception as e:
                    print('visdom error:', e)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann) for ann in targets]

        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = cfg['train_cfg']['loc_weight'] * loss_l + cfg['train_cfg']['cls_weight'] * loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('iter ' + repr(iteration) +
                  '\t|| loc/conf/all: %.4f / %.4f / %.4f ||' % (loss_l.item(), loss_c.item(), loss.item()),
                  end=' ')
            print('timer: %.4f sec.' % (t1 - t0))

        if args.visdom:
            try:
                update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                                iter_plot, epoch_plot, 'append')
            except Exception as e:
                print('visdom error:', e)

        if iteration in dataset_cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), args.save_folder + 'FlashNet_' +
                       dataset.name + '_' + repr(iteration) + '.pth')

    torch.save(net.state_dict(), args.save_folder + 'FlashNet_' + dataset.name + '.pth')


"""
Utility Functions Implement
"""


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


"""
    main
"""
if __name__ == '__main__':
    train()

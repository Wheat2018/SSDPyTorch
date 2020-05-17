"""
    By.Wheat
    2020.05.17
"""
import os
import torch
import torch.nn as nn
import copy
from collections import defaultdict


class SSDBackbone(nn.Module):
    """
    Single Shot Multibox Architecture
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Note: SSDBase refers to the part of SSD that is independent of the predictor
    """

    def __init__(self, base, is_cuda=torch.cuda.is_available()):
        super(SSDBackbone, self).__init__()
        # configure information
        self.is_cuda = is_cuda

        # block layers recorder
        # --Note: The elements in it should only be used for comparing. (isinstance)
        self.blocks = []
        self.detect_blocks = []

        # SSD backbone network
        # --base layers
        self.base = base
        self.blocks += self.base.blocks
        self.detect_blocks += self.base.detect_blocks
        # --extra layers
        self.extras = nn.ModuleList()
        self.extra_net()

        self.name = type(self).__name__

    def extra_net(self):
        in_channels = self.base.out_channels
        # block1---------------------
        out, in_channels = repeat_conv2d(1, in_channels, 256, 1,
                                         activation=nn.ReLU(inplace=True))
        self.extras += out
        out, in_channels = repeat_conv2d(1, in_channels, 512, 3, stride=2, padding=1,
                                         activation=nn.ReLU(inplace=True))
        self.extras += out
        self.blocks += [(self.extras[-1], in_channels)]  # block1
        self.detect_blocks += [(self.extras[-1], in_channels)] # detect this block

        # block2---------------------
        out, in_channels = repeat_conv2d(1, in_channels, 128, 1,
                                         activation=nn.ReLU(inplace=True))
        self.extras += out
        out, in_channels = repeat_conv2d(1, in_channels, 256, 3, stride=2, padding=1,
                                         activation=nn.ReLU(inplace=True))
        self.extras += out
        self.blocks += [(self.extras[-1], in_channels)]  # block2
        self.detect_blocks += [(self.extras[-1], in_channels)] # detect this block

        # block3---------------------
        out, in_channels = repeat_conv2d(1, in_channels, 128, 1,
                                         activation=nn.ReLU(inplace=True))
        self.extras += out
        out, in_channels = repeat_conv2d(1, in_channels, 256, 3,
                                         activation=nn.ReLU(inplace=True))
        self.extras += out
        self.blocks += [(self.extras[-1], in_channels)]  # block3
        self.detect_blocks += [(self.extras[-1], in_channels)] # detect this block

        # block4---------------------
        out, in_channels = repeat_conv2d(1, in_channels, 128, 1,
                                         activation=nn.ReLU(inplace=True))
        self.extras += out
        out, in_channels = repeat_conv2d(1, in_channels, 256, 3,
                                         activation=nn.ReLU(inplace=True))
        self.extras += out
        self.blocks += [(self.extras[-1], in_channels)]  # block4
        self.detect_blocks += [(self.extras[-1], in_channels)]  # detect this block

    def convolution_predictor(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def forward_for_detect_source(self, x):
        sources = []

        block_idx = 0
        # apply base net
        for layer in self.base:
            x = layer(x)
            if layer is self.detect_blocks[block_idx][0]:
                sources.append(x)
                block_idx += 1

        # apply extra layers and cache source layer outputs
        for layer in self.extras:
            x = layer(x)
            if layer is self.detect_blocks[block_idx][0]:
                sources.append(x)
                block_idx += 1

        return sources

    def detect(self, *args):
        raise NotImplementedError

    @staticmethod
    def analyse_state_dict(state_dict):
        result = '{'
        counts = defaultdict(int)
        for para_name, para_value in state_dict.items():
            counts[para_name.split('.')[0]] += 1
        for para_name, count in counts.items():
            result += '%s: %d, ' % (para_name, count)

        return result + '}'

    @staticmethod
    def load_weights(module, file):
        other, ext = os.path.splitext(file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            print(file)
            try:
                ssd_weights = torch.load(file, map_location=torch.device('cuda' if module.is_cuda else 'cpu'))
            except Exception as e:
                print('None file or error file.', e)
                return
            model_dict = module.state_dict()
            ssd_weights = {k2: v1 for (k1, v1), (k2, v2)
                           in zip(ssd_weights.items(), model_dict.items())
                           if v2.shape == v1.shape}
            model_dict.update(ssd_weights)
            # model_dict_items = sorted(model_dict.items(), key=lambda item: (item[0].split('.')[0].replace('vgg', 'base') +
            #                            item[0].split('.')[1].zfill(3)))
            # ssd_weights_items = sorted(ssd_weights.items(), key=lambda item: (item[0].split('.')[0].replace('vgg', 'base') +
            #                            item[0].split('.')[1].zfill(3)))
            # for (k1, v1), (k2, v2) in zip(ssd_weights_items, model_dict_items):
            #     model_dict[k2] = v1
            print('update following parameters:', SSDBackbone.analyse_state_dict(ssd_weights))
            print('Net parameters:', SSDBackbone.analyse_state_dict(model_dict))
            print('Net has %d parameters. Loaded %d parameters' % (len(model_dict), len(ssd_weights)))
            if len(ssd_weights) < len(model_dict):
                print('Init remain parameters.')
                module.apply(SSDBackbone.weights_init)
            module.load_state_dict(model_dict)
            print('Finished!')
        else:
            print('Only .pth and .pkl files supported.')

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight.data)
            m.bias.data.zero_()


def repeat_conv2d(repeat, in_channels, out_channels, kernel_size, stride=1,
                  padding=0, dilation=1, groups=1, bias=True,
                  bn=False, activation=None):
    out_list = []
    for i in range(repeat):
        in_channels = in_channels if i == 0 else out_channels
        out_list += [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding, dilation, groups, bias)]
        if bn:
            out_list += [nn.BatchNorm2d(out_channels)]
        if activation is not None:
            out_list += [copy.deepcopy(activation)]
    return out_list, out_channels


class VGG(nn.ModuleList):
    def __init__(self, in_channels, batch_norm=False):
        super(VGG, self).__init__()
        self.blocks = []

        # block1---------------------
        out, in_channels = repeat_conv2d(2, in_channels, 64, 3, padding=1, bn=batch_norm,
                                         activation=nn.ReLU(inplace=True))
        self += out
        self.blocks += [(self[-1], in_channels)]   # block1
        self += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # block2---------------------
        out, in_channels = repeat_conv2d(2, in_channels, 128, 3, padding=1, bn=batch_norm,
                                         activation=nn.ReLU(inplace=True))
        self += out
        self.blocks += [(self[-1], in_channels)]   # block2
        self += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # block3---------------------
        out, in_channels = repeat_conv2d(3, in_channels, 256, 3, padding=1, bn=batch_norm,
                                         activation=nn.ReLU(inplace=True))
        self += out
        self.blocks += [(self[-1], in_channels)]   # block3
        self += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]

        # block4---------------------
        out, in_channels = repeat_conv2d(3, in_channels, 512, 3, padding=1, bn=batch_norm,
                                         activation=nn.ReLU(inplace=True))
        self += out
        self.blocks += [(self[-1], in_channels)]   # block4
        self += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # block5---------------------
        out, in_channels = repeat_conv2d(3, in_channels, 512, 3, padding=1, bn=batch_norm,
                                         activation=nn.ReLU(inplace=True))
        self += out
        self.blocks += [(self[-1], in_channels)]   # block5
        self += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]

        # block6---------------------
        out, in_channels = repeat_conv2d(1, in_channels, 1024, 3, padding=6, dilation=6, bn=batch_norm,
                                         activation=nn.ReLU(inplace=True))
        self += out
        self.blocks += [(self[-1], in_channels)]   # block6

        # block7---------------------
        out, in_channels = repeat_conv2d(1, in_channels, 1024, 1, bn=batch_norm,
                                         activation=nn.ReLU(inplace=True))
        self += out
        self.blocks += [(self[-1], in_channels)]   # block7

        self.out_channels = in_channels
        self.detect_blocks = [self.blocks[3], self.blocks[6]]

    def forward(self):
        raise NotImplementedError

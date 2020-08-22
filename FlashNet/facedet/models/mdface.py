# from . import backbones as backbones_mod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.operators
import math
import os
from ...facedet.models.common_ops import *
# from .backbones.mnasnet import _InvertedResidual, _BN_MOMENTUM
# from .backbones.se_module import SELayer
# from .cos_attn import Cos_Attn
# from .lrn_layer import LocalRelationalLayer

__all__ = ['MDFace_light', 'MDFace', 'MDFace_9_anchor', 'MDFace_2x', 'MDFace_3x', 'MDFace_3x_share_head', 'MDFace_share_head']

class Cos_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self):
        super(Cos_Attn, self).__init__()
        # self.chanel_in = in_dim
        # self.activation = activation
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, N = x.size()
        proj_query = x.view(m_batchsize, -1, N).permute(0, 2, 1)  # B X N X C
        proj_key = x.view(m_batchsize, -1, N)  # B X C x N
        q_norm = proj_query.norm(2, dim=2)  # B X N X 1
        nm = torch.bmm(q_norm.view(m_batchsize, N, 1), q_norm.view(m_batchsize, 1, N))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / nm
        return norm_energy

class MDFace_light(nn.Module):

    def __init__(self, phase, cfg):
        super(MDFace_light, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.out_featmaps = cfg['out_featmaps']
        self.feat_adp = cfg['feat_adp']
        self.conv1 = conv_bn(3, 8, 2)
        # self.conv1 = conv_bn_5X5(3, 12, 2)
        # self.conv2 = conv_bn(12, 12, 2)
        # self.conv1 = hetconv_bn_5X5(3, 12, num_groups=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.p3_conv1 = InvertedResidual(inp=8, oup=24, kernel=3, stride=2, expand_ratio=4)
        self.p3_conv2 = InvertedResidual(inp=24, oup=24, kernel=5, stride=1, expand_ratio=4)
        self.p3_conv3 = InvertedResidual(inp=24, oup=32, kernel=5, stride=1, expand_ratio=4)

        self.p4_conv1 = InvertedResidual(inp=32, oup=48, kernel=3, stride=2, expand_ratio=4)
        self.p4_conv2 = InvertedResidual(inp=48, oup=48, kernel=5, stride=1, expand_ratio=3)
        self.p4_conv3 = InvertedResidual(inp=48, oup=48, kernel=3, stride=1, expand_ratio=3)

        self.p5_conv1 = InvertedResidual(inp=48, oup=64, kernel=3, stride=2, expand_ratio=3)
        self.p5_conv2 = InvertedResidual(inp=64, oup=64, kernel=5, stride=1, expand_ratio=3)
        self.p5_conv3 = InvertedResidual(inp=64, oup=64, kernel=3, stride=1, expand_ratio=3)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(48, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        if self.feat_adp:
            self.p3_adp = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
            self.p4_adp = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
            self.p5_adp = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

        self.box_head, self.cls_head = self.multibox(self.num_classes)
        self.attn = Cos_Attn()
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []

        channels = [32, 32, 32]
        loc_layers += [InvertedResidual_Head(inp=channels[0], oup=2 * 4, kernel=3, stride=1, expand_ratio=1)]
        conf_layers += [InvertedResidual_Head(inp=channels[0], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=1)]
        loc_layers += [InvertedResidual_Head(inp=channels[1], oup=2 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [
            InvertedResidual_Head(inp=channels[1], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        loc_layers += [InvertedResidual_Head(inp=channels[2], oup=2 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [
            InvertedResidual_Head(inp=channels[2], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def _upsample_add(self, x, y):
        size = [v for v in y.size()[2:]]
        size = [int(i) for i in size]
        return F.interpolate(x, size=size, mode='nearest') + y

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            weights = {k: v for k, v in chk['state_dict'].items() if k not in ignored}
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbones(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)

            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)
            self.ldmk_head.apply(initialize_layer)

    def forward(self, x):

        features = list()
        attentions = list()
        loc = list()
        conf = list()
        ldmk = list()
        attn = list()

        x = self.conv1(x)
        # s4 = self.conv2(x)
        s4 = self.maxpool1(x)

        x = self.p3_conv1(s4)
        x = self.p3_conv2(x)
        s8 = self.p3_conv3(x)

        x = self.p4_conv1(s8)
        x = self.p4_conv2(x)
        s16 = self.p4_conv3(x)

        x = self.p5_conv1(s16)
        x = self.p5_conv2(x)
        s32 = self.p5_conv3(x)

        p5 = self.latlayer1(s32)
        p4 = self._upsample_add(p5, self.latlayer2(s16))
        p3 = self._upsample_add(p4, self.latlayer3(s8))

        features.append(p3)
        features.append(p4)
        features.append(p5)

        if self.feat_adp:
            # print('self.p3_adp(p3).size()',self.p3_adp(p3).size())
            # print(self.p3_adp(p3).pow(2).mean(1).unsqueeze(1).size())
            attentions.append(self.p3_adp(p3))
            attentions.append(self.p4_adp(p4))
            attentions.append(self.p5_adp(p5))
        else:
            attentions.append(p3.pow(2).mean(1).unsqueeze(1))
            attentions.append(p4.pow(2).mean(1).unsqueeze(1))
            attentions.append(p5.pow(2).mean(1).unsqueeze(1))

        for idx, t in enumerate(features):
            # print(idx)
            conf.append(self.cls_head[idx](t).permute(0, 2, 3, 1).contiguous())
            loc.append(self.box_head[idx](t).permute(0, 2, 3, 1).contiguous())
            # ldmk.append(self.ldmk_head[idx](t).permute(0, 2, 3, 1).contiguous())
            # attn.append(self.attn(t))

        # print(attn[0].size())
        # print(attn[1].size())
        # print(attn[2].size())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # features = torch.cat([o.view(o.size(0), o.size(1), -1) for o in features], -1).unsqueeze(-1)
        # features = torch.cat([features, features], -1)
        # features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)
        # print(features.size())
        # ldmk = torch.cat([o.view(o.size(0), -1) for o in ldmk], 1)

        if self.phase == "test":
            if self.num_classes == 2:
                output = (loc.view(loc.size(0), -1, 4),
                          self.softmax(conf.view(-1, self.num_classes))
                          )

            elif self.num_classes == 1:  # focal loss
                output = (loc.view(loc.size(0), -1, 4),
                          conf.view(-1, self.num_classes).sigmoid().max(1))

        else:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes))

        if self.out_featmaps:
            return output, attentions
        else:
            return output

class MDFace(nn.Module):

    def __init__(self, phase, cfg):
        super(MDFace, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.out_featmaps = cfg['out_featmaps']
        self.feat_adp = cfg['feat_adp']
        # self.conv1 = conv_bn(3, 12, 2)
        self.conv1 = conv_bn_5X5(3, 12, 2)
        # self.conv2 = conv_bn(12, 12, 2)
        # self.conv1 = hetconv_bn_5X5(3, 12, num_groups=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.p3_conv1 = InvertedResidual(inp=12, oup=32, kernel=3, stride=2, expand_ratio=3)
        self.p3_conv2 = InvertedResidual(inp=32, oup=32, kernel=5, stride=1, expand_ratio=3)
        self.p3_conv3 = InvertedResidual(inp=32, oup=32, kernel=5, stride=1, expand_ratio=3)

        self.p4_conv1 = InvertedResidual(inp=32, oup=48, kernel=3, stride=2, expand_ratio=3)
        self.p4_conv2 = InvertedResidual(inp=48, oup=48, kernel=5, stride=1, expand_ratio=3)
        self.p4_conv3 = InvertedResidual(inp=48, oup=48, kernel=3, stride=1, expand_ratio=3)

        self.p5_conv1 = InvertedResidual(inp=48, oup=64, kernel=3, stride=2, expand_ratio=3)
        self.p5_conv2 = InvertedResidual(inp=64, oup=64, kernel=5, stride=1, expand_ratio=3)
        self.p5_conv3 = InvertedResidual(inp=64, oup=64, kernel=3, stride=1, expand_ratio=3)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(48, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        if self.feat_adp:
            self.p3_adp = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
            self.p4_adp = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
            self.p5_adp = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

        self.box_head, self.cls_head = self.multibox(self.num_classes)
        self.attn = Cos_Attn()
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []

        channels = [32, 32, 32]
        loc_layers += [InvertedResidual_Head(inp=channels[0], oup=2 * 4, kernel=3, stride=1, expand_ratio=1)]
        conf_layers += [InvertedResidual_Head(inp=channels[0], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=1)]
        loc_layers += [InvertedResidual_Head(inp=channels[1], oup=2 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[1], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        loc_layers += [InvertedResidual_Head(inp=channels[2], oup=2 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[2], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def _upsample_add(self, x, y):
        size = [v for v in y.size()[2:]]
        size = [int(i) for i in size]
        return F.interpolate(x, size=size, mode='nearest') + y

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            weights = { k: v for k, v in chk['state_dict'].items() if k not in ignored }
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbones(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)
            self.ldmk_head.apply(initialize_layer)

    def forward(self, x):

        features = list()
        attentions = list()
        loc = list()
        conf = list()
        ldmk = list()
        attn = list()


        x = self.conv1(x)
        # s4 = self.conv2(x)
        s4 = self.maxpool1(x)

        x = self.p3_conv1(s4)
        x = self.p3_conv2(x)
        s8 = self.p3_conv3(x)

        x = self.p4_conv1(s8)
        x = self.p4_conv2(x)
        s16 = self.p4_conv3(x)

        x = self.p5_conv1(s16)
        x = self.p5_conv2(x)
        s32 = self.p5_conv3(x)

        p5 = self.latlayer1(s32)
        p4 = self._upsample_add(p5, self.latlayer2(s16))
        p3 = self._upsample_add(p4, self.latlayer3(s8))

        features.append(p3)
        features.append(p4)
        features.append(p5)

        if self.feat_adp:
            attentions.append(self.p3_adp(p3).pow(2).mean(1).unsqueeze(1))
            attentions.append(self.p4_adp(p4).pow(2).mean(1).unsqueeze(1))
            attentions.append(self.p5_adp(p5).pow(2).mean(1).unsqueeze(1))
        else:
            attentions.append(p3.pow(2).mean(1).unsqueeze(1))
            attentions.append(p4.pow(2).mean(1).unsqueeze(1))
            attentions.append(p5.pow(2).mean(1).unsqueeze(1))

        for idx, t in enumerate(features):
            # print(idx)
            conf.append(self.cls_head[idx](t).permute(0, 2, 3, 1).contiguous())
            loc.append(self.box_head[idx](t).permute(0, 2, 3, 1).contiguous())
            # ldmk.append(self.ldmk_head[idx](t).permute(0, 2, 3, 1).contiguous())
            # attn.append(self.attn(t))

        # print(attn[0].size())
        # print(attn[1].size())
        # print(attn[2].size())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)


        # features = torch.cat([o.view(o.size(0), o.size(1), -1) for o in features], -1).unsqueeze(-1)
        # features = torch.cat([features, features], -1)
        # features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)


        # print(features.size())
        # ldmk = torch.cat([o.view(o.size(0), -1) for o in ldmk], 1)

        if self.phase == "test":
            if self.num_classes == 2:
                output = (loc.view(loc.size(0), -1, 4),
                          self.softmax(conf.view(-1, self.num_classes))
                          )

            elif self.num_classes == 1:  # focal loss
                output = (loc.view(loc.size(0), -1, 4),
                          conf.view(-1, self.num_classes).sigmoid().max(1))

        else:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes))

        if self.out_featmaps:
            return output, attentions
        else:
            return output

class MDFace_9_anchor(nn.Module):

    def __init__(self, phase, cfg):
        super(MDFace_9_anchor, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.out_featmaps = cfg['out_featmaps']
        # self.conv1 = conv_bn(3, 12, 2)
        self.conv1 = conv_bn_5X5(3, 12, 2)
        # self.conv2 = conv_bn(12, 12, 2)
        # self.conv1 = hetconv_bn_5X5(3, 12, num_groups=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.p3_conv1 = InvertedResidual(inp=12, oup=32, kernel=3, stride=2, expand_ratio=3)
        self.p3_conv2 = InvertedResidual(inp=32, oup=32, kernel=5, stride=1, expand_ratio=3)
        self.p3_conv3 = InvertedResidual(inp=32, oup=32, kernel=5, stride=1, expand_ratio=3)

        self.p4_conv1 = InvertedResidual(inp=32, oup=48, kernel=3, stride=2, expand_ratio=3)
        self.p4_conv2 = InvertedResidual(inp=48, oup=48, kernel=5, stride=1, expand_ratio=3)
        self.p4_conv3 = InvertedResidual(inp=48, oup=48, kernel=3, stride=1, expand_ratio=3)

        self.p5_conv1 = InvertedResidual(inp=48, oup=64, kernel=3, stride=2, expand_ratio=3)
        self.p5_conv2 = InvertedResidual(inp=64, oup=64, kernel=5, stride=1, expand_ratio=3)
        self.p5_conv3 = InvertedResidual(inp=64, oup=64, kernel=3, stride=1, expand_ratio=3)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(48, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.box_head, self.cls_head = self.multibox(self.num_classes)
        self.attn = Cos_Attn()
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []

        channels = [32, 32, 32]
        loc_layers += [InvertedResidual_Head(inp=channels[0], oup=3 * 4, kernel=3, stride=1, expand_ratio=1)]
        conf_layers += [InvertedResidual_Head(inp=channels[0], oup=3 * num_classes, kernel=3, stride=1, expand_ratio=1)]
        loc_layers += [InvertedResidual_Head(inp=channels[1], oup=3 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[1], oup=3 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        loc_layers += [InvertedResidual_Head(inp=channels[2], oup=3 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[2], oup=3 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def _upsample_add(self, x, y):
        size = [v for v in y.size()[2:]]
        size = [int(i) for i in size]
        return F.interpolate(x, size=size, mode='nearest') + y

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            weights = { k: v for k, v in chk['state_dict'].items() if k not in ignored }
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbones(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)
            self.ldmk_head.apply(initialize_layer)

    def forward(self, x):

        features = list()
        loc = list()
        conf = list()
        ldmk = list()
        attn = list()


        x = self.conv1(x)
        # s4 = self.conv2(x)
        s4 = self.maxpool1(x)

        x = self.p3_conv1(s4)
        x = self.p3_conv2(x)
        s8 = self.p3_conv3(x)

        x = self.p4_conv1(s8)
        x = self.p4_conv2(x)
        s16 = self.p4_conv3(x)

        x = self.p5_conv1(s16)
        x = self.p5_conv2(x)
        s32 = self.p5_conv3(x)

        p5 = self.latlayer1(s32)
        p4 = self._upsample_add(p5, self.latlayer2(s16))
        p3 = self._upsample_add(p4, self.latlayer3(s8))

        features.append(p3)
        features.append(p4)
        features.append(p5)

        for idx, t in enumerate(features):
            # print(idx)
            conf.append(self.cls_head[idx](t).permute(0, 2, 3, 1).contiguous())
            loc.append(self.box_head[idx](t).permute(0, 2, 3, 1).contiguous())
            # ldmk.append(self.ldmk_head[idx](t).permute(0, 2, 3, 1).contiguous())
            # attn.append(self.attn(t))

        # print(attn[0].size())
        # print(attn[1].size())
        # print(attn[2].size())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)


        features = torch.cat([o.view(o.size(0), o.size(1), -1) for o in features], -1).unsqueeze(-1)
        features = torch.cat([features, features], -1)
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)
        # print(features.size())
        # ldmk = torch.cat([o.view(o.size(0), -1) for o in ldmk], 1)

        if self.phase == "test":
            if self.num_classes == 2:
                output = (loc.view(loc.size(0), -1, 4),
                          self.softmax(conf.view(-1, self.num_classes))
                          )

            elif self.num_classes == 1:  # focal loss
                output = (loc.view(loc.size(0), -1, 4),
                          conf.view(-1, self.num_classes).sigmoid().max(1))

        else:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes))

        if self.out_featmaps:
            return output, features
        else:
            return output

class MDFace_2x(nn.Module):

    def __init__(self, phase, cfg):
        super(MDFace_2x, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.out_featmaps = cfg['out_featmaps']
        # self.conv1 = conv_bn(3, 12, 2)
        self.conv1 = conv_bn_5X5(3, 24, 2)
        # self.conv1 = hetconv_bn_5X5(3, 12, num_groups=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.p3_conv1 = InvertedResidual(inp=24, oup=64, kernel=3, stride=2, expand_ratio=3)
        self.p3_conv2 = InvertedResidual(inp=64, oup=64, kernel=5, stride=1, expand_ratio=3)
        self.p3_conv3 = InvertedResidual(inp=64, oup=64, kernel=5, stride=1, expand_ratio=3)

        self.p4_conv1 = InvertedResidual(inp=64, oup=96, kernel=3, stride=2, expand_ratio=3)
        self.p4_conv2 = InvertedResidual(inp=96, oup=96, kernel=5, stride=1, expand_ratio=3)
        self.p4_conv3 = InvertedResidual(inp=96, oup=96, kernel=3, stride=1, expand_ratio=3)

        self.p5_conv1 = InvertedResidual(inp=96, oup=128, kernel=3, stride=2, expand_ratio=3)
        self.p5_conv2 = InvertedResidual(inp=128, oup=128, kernel=5, stride=1, expand_ratio=3)
        self.p5_conv3 = InvertedResidual(inp=128, oup=128, kernel=3, stride=1, expand_ratio=3)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        self.box_head, self.cls_head = self.multibox(self.num_classes)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []

        channels = [64, 64, 64]
        loc_layers += [InvertedResidual_Head(inp=channels[0], oup=2 * 4, kernel=3, stride=1, expand_ratio=1)]
        conf_layers += [InvertedResidual_Head(inp=channels[0], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=1)]
        loc_layers += [InvertedResidual_Head(inp=channels[1], oup=2 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[1], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        loc_layers += [InvertedResidual_Head(inp=channels[2], oup=2 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[2], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def _upsample_add(self, x, y):
        size = [v for v in y.size()[2:]]
        size = [int(i) for i in size]
        return F.interpolate(x, size=size, mode='nearest') + y

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            weights = { k: v for k, v in chk['state_dict'].items() if k not in ignored }
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbones(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)
            self.ldmk_head.apply(initialize_layer)

    def forward(self, x):

        features = list()
        attentions = list()
        loc = list()
        conf = list()
        ldmk = list()

        x = self.conv1(x)
        s4 = self.maxpool1(x)

        x = self.p3_conv1(s4)
        x = self.p3_conv2(x)
        s8 = self.p3_conv3(x)

        x = self.p4_conv1(s8)
        x = self.p4_conv2(x)
        s16 = self.p4_conv3(x)

        x = self.p5_conv1(s16)
        x = self.p5_conv2(x)
        s32 = self.p5_conv3(x)

        p5 = self.latlayer1(s32)
        p4 = self._upsample_add(p5, self.latlayer2(s16))
        p3 = self._upsample_add(p4, self.latlayer3(s8))

        features.append(p3)
        features.append(p4)
        features.append(p5)

        attentions.append(p3.pow(2).mean(1).unsqueeze(1))
        attentions.append(p4.pow(2).mean(1).unsqueeze(1))
        attentions.append(p5.pow(2).mean(1).unsqueeze(1))

        for idx, t in enumerate(features):
            # print(idx)
            conf.append(self.cls_head[idx](t).permute(0, 2, 3, 1).contiguous())
            loc.append(self.box_head[idx](t).permute(0, 2, 3, 1).contiguous())
            # ldmk.append(self.ldmk_head[idx](t).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # ldmk = torch.cat([o.view(o.size(0), -1) for o in ldmk], 1)
        # features = torch.cat([o.view(o.size(0), o.size(1), -1) for o in features], -1).unsqueeze(-1)
        # features = torch.cat([features, features], -1)
        # features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)
        # print(features.size())

        if self.phase == "test":
            if self.num_classes == 2:
                output = (loc.view(loc.size(0), -1, 4),
                          self.softmax(conf.view(-1, self.num_classes))
                          )

            elif self.num_classes == 1:  # focal loss
                output = (loc.view(loc.size(0), -1, 4),
                          conf.view(-1, self.num_classes).sigmoid().max(1))

        else:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes))

        if self.out_featmaps:
            return output, attentions
        else:
            return output

class MDFace_3x(nn.Module):

    def __init__(self, phase, cfg):
        super(MDFace_3x, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.out_featmaps = cfg['out_featmaps']
        # self.conv1 = conv_bn(3, 12, 2)
        self.conv1 = conv_bn_5X5(3, 36, 2)
        # self.conv1 = hetconv_bn_5X5(3, 12, num_groups=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.p3_conv1 = InvertedResidual(inp=36, oup=96, kernel=3, stride=2, expand_ratio=3)
        self.p3_conv2 = InvertedResidual(inp=96, oup=96, kernel=5, stride=1, expand_ratio=3)
        self.p3_conv3 = InvertedResidual(inp=96, oup=96, kernel=5, stride=1, expand_ratio=3)

        self.p4_conv1 = InvertedResidual(inp=96, oup=144, kernel=3, stride=2, expand_ratio=3)
        self.p4_conv2 = InvertedResidual(inp=144, oup=144, kernel=5, stride=1, expand_ratio=3)
        self.p4_conv3 = InvertedResidual(inp=144, oup=144, kernel=3, stride=1, expand_ratio=3)

        self.p5_conv1 = InvertedResidual(inp=144, oup=192, kernel=3, stride=2, expand_ratio=3)
        self.p5_conv2 = InvertedResidual(inp=192, oup=192, kernel=5, stride=1, expand_ratio=3)
        self.p5_conv3 = InvertedResidual(inp=192, oup=192, kernel=3, stride=1, expand_ratio=3)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(144, 96, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0)

        self.box_head, self.cls_head = self.multibox(self.num_classes)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []

        channels = [96, 96, 96]
        loc_layers += [InvertedResidual_Head(inp=channels[0], oup=2 * 4, kernel=3, stride=1, expand_ratio=1)]
        conf_layers += [InvertedResidual_Head(inp=channels[0], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=1)]
        loc_layers += [InvertedResidual_Head(inp=channels[1], oup=2 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[1], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        loc_layers += [InvertedResidual_Head(inp=channels[2], oup=2 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[2], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def _upsample_add(self, x, y):
        size = [v for v in y.size()[2:]]
        size = [int(i) for i in size]
        return F.interpolate(x, size=size, mode='nearest') + y

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            weights = { k: v for k, v in chk['state_dict'].items() if k not in ignored }
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbones(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)
            self.ldmk_head.apply(initialize_layer)

    def forward(self, x):

        features = list()
        attentions = list()
        loc = list()
        conf = list()
        ldmk = list()

        x = self.conv1(x)
        s4 = self.maxpool1(x)

        x = self.p3_conv1(s4)
        x = self.p3_conv2(x)
        s8 = self.p3_conv3(x)

        x = self.p4_conv1(s8)
        x = self.p4_conv2(x)
        s16 = self.p4_conv3(x)

        x = self.p5_conv1(s16)
        x = self.p5_conv2(x)
        s32 = self.p5_conv3(x)

        p5 = self.latlayer1(s32)
        p4 = self._upsample_add(p5, self.latlayer2(s16))
        p3 = self._upsample_add(p4, self.latlayer3(s8))

        features.append(p3)
        features.append(p4)
        features.append(p5)

        attentions.append(p3.pow(2).mean(1).unsqueeze(1))
        attentions.append(p4.pow(2).mean(1).unsqueeze(1))
        attentions.append(p5.pow(2).mean(1).unsqueeze(1))

        for idx, t in enumerate(features):
            # print(idx)
            conf.append(self.cls_head[idx](t).permute(0, 2, 3, 1).contiguous())
            loc.append(self.box_head[idx](t).permute(0, 2, 3, 1).contiguous())
            # ldmk.append(self.ldmk_head[idx](t).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # ldmk = torch.cat([o.view(o.size(0), -1) for o in ldmk], 1)
        # features = torch.cat([o.view(o.size(0), o.size(1), -1) for o in features], -1).unsqueeze(-1)
        # features = torch.cat([features, features], -1)
        # features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)

        if self.phase == "test":
            if self.num_classes == 2:
                output = (loc.view(loc.size(0), -1, 4),
                          self.softmax(conf.view(-1, self.num_classes))
                          )

            elif self.num_classes == 1:  # focal loss
                output = (loc.view(loc.size(0), -1, 4),
                          conf.view(-1, self.num_classes).sigmoid().max(1))

        else:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes))

        if self.out_featmaps:
            return output, attentions
        else:
            return output

class MDFace_3x_share_head(nn.Module):

    def __init__(self, phase, cfg):
        super(MDFace_3x_share_head, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.out_featmaps = cfg['out_featmaps']
        # self.conv1 = conv_bn(3, 12, 2)
        self.conv1 = conv_bn_5X5(3, 36, 2)
        # self.conv1 = hetconv_bn_5X5(3, 12, num_groups=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.p3_conv1 = InvertedResidual(inp=36, oup=96, kernel=3, stride=2, expand_ratio=3)
        self.p3_conv2 = InvertedResidual(inp=96, oup=96, kernel=5, stride=1, expand_ratio=3)
        self.p3_conv3 = InvertedResidual(inp=96, oup=96, kernel=5, stride=1, expand_ratio=3)

        self.p4_conv1 = InvertedResidual(inp=96, oup=144, kernel=3, stride=2, expand_ratio=3)
        self.p4_conv2 = InvertedResidual(inp=144, oup=144, kernel=5, stride=1, expand_ratio=3)
        self.p4_conv3 = InvertedResidual(inp=144, oup=144, kernel=3, stride=1, expand_ratio=3)

        self.p5_conv1 = InvertedResidual(inp=144, oup=192, kernel=3, stride=2, expand_ratio=3)
        self.p5_conv2 = InvertedResidual(inp=192, oup=192, kernel=5, stride=1, expand_ratio=3)
        self.p5_conv3 = InvertedResidual(inp=192, oup=192, kernel=3, stride=1, expand_ratio=3)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(192, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(144, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0)

        self.box_head, self.cls_head = self.multibox(self.num_classes)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []

        channels = [32, 32, 32]
        loc_layers += [InvertedResidual_Head(inp=channels[0], oup=2 * 4, kernel=3, stride=1, expand_ratio=1)]
        conf_layers += [InvertedResidual_Head(inp=channels[0], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=1)]
        loc_layers += [InvertedResidual_Head(inp=channels[1], oup=2 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[1], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        loc_layers += [InvertedResidual_Head(inp=channels[2], oup=2 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[2], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def _upsample_add(self, x, y):
        size = [v for v in y.size()[2:]]
        size = [int(i) for i in size]
        return F.interpolate(x, size=size, mode='nearest') + y

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            weights = { k: v for k, v in chk['state_dict'].items() if k not in ignored }
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbones(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)
            self.ldmk_head.apply(initialize_layer)

    def forward(self, x):

        features = list()
        loc = list()
        conf = list()
        ldmk = list()

        x = self.conv1(x)
        s4 = self.maxpool1(x)

        x = self.p3_conv1(s4)
        x = self.p3_conv2(x)
        s8 = self.p3_conv3(x)

        x = self.p4_conv1(s8)
        x = self.p4_conv2(x)
        s16 = self.p4_conv3(x)

        x = self.p5_conv1(s16)
        x = self.p5_conv2(x)
        s32 = self.p5_conv3(x)

        p5 = self.latlayer1(s32)
        p4 = self._upsample_add(p5, self.latlayer2(s16))
        p3 = self._upsample_add(p4, self.latlayer3(s8))

        features.append(p3)
        features.append(p4)
        features.append(p5)

        for idx, t in enumerate(features):
            # print(idx)
            conf.append(self.cls_head[idx](t).permute(0, 2, 3, 1).contiguous())
            loc.append(self.box_head[idx](t).permute(0, 2, 3, 1).contiguous())
            # ldmk.append(self.ldmk_head[idx](t).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # ldmk = torch.cat([o.view(o.size(0), -1) for o in ldmk], 1)

        if self.phase == "test":
            if self.num_classes == 2:
                output = (loc.view(loc.size(0), -1, 4),
                          self.softmax(conf.view(-1, self.num_classes))
                          )

            elif self.num_classes == 1:  # focal loss
                output = (loc.view(loc.size(0), -1, 4),
                          conf.view(-1, self.num_classes).sigmoid().max(1))

        else:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes))

        if self.out_featmaps:
            return output, features
        else:
            return output

class MDFace_share_head(nn.Module):

    def __init__(self, phase, cfg):
        super(MDFace_share_head, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.out_featmaps = cfg['out_featmaps']
        # self.conv1 = conv_bn(3, 12, 2)
        self.conv1 = conv_bn_5X5(3, 12, 2)
        # self.conv2 = conv_bn(12, 12, 2)
        # self.conv1 = hetconv_bn_5X5(3, 12, num_groups=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.p3_conv1 = InvertedResidual(inp=12, oup=32, kernel=3, stride=2, expand_ratio=3)
        self.p3_conv2 = InvertedResidual(inp=32, oup=32, kernel=5, stride=1, expand_ratio=3)
        self.p3_conv3 = InvertedResidual(inp=32, oup=32, kernel=5, stride=1, expand_ratio=3)

        self.p4_conv1 = InvertedResidual(inp=32, oup=48, kernel=3, stride=2, expand_ratio=3)
        self.p4_conv2 = InvertedResidual(inp=48, oup=48, kernel=5, stride=1, expand_ratio=3)
        self.p4_conv3 = InvertedResidual(inp=48, oup=48, kernel=3, stride=1, expand_ratio=3)

        self.p5_conv1 = InvertedResidual(inp=48, oup=64, kernel=3, stride=2, expand_ratio=3)
        self.p5_conv2 = InvertedResidual(inp=64, oup=64, kernel=5, stride=1, expand_ratio=3)
        self.p5_conv3 = InvertedResidual(inp=64, oup=64, kernel=3, stride=1, expand_ratio=3)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(64, 96, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(32, 96, kernel_size=1, stride=1, padding=0)

        self.box_head, self.cls_head = self.multibox(self.num_classes)
        self.attn = Cos_Attn()
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []

        channels = [96, 96, 96]
        loc_layers += [InvertedResidual_Head(inp=channels[0], oup=2 * 4, kernel=3, stride=1, expand_ratio=1)]
        conf_layers += [InvertedResidual_Head(inp=channels[0], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=1)]
        loc_layers += [InvertedResidual_Head(inp=channels[1], oup=2 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[1], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        loc_layers += [InvertedResidual_Head(inp=channels[2], oup=2 * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[2], oup=2 * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def _upsample_add(self, x, y):
        size = [v for v in y.size()[2:]]
        size = [int(i) for i in size]
        return F.interpolate(x, size=size, mode='nearest') + y

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            weights = { k: v for k, v in chk['state_dict'].items() if k not in ignored }
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbones(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)
            self.ldmk_head.apply(initialize_layer)

    def forward(self, x):

        features = list()
        loc = list()
        conf = list()
        ldmk = list()
        attn = list()


        x = self.conv1(x)
        # s4 = self.conv2(x)
        s4 = self.maxpool1(x)

        x = self.p3_conv1(s4)
        x = self.p3_conv2(x)
        s8 = self.p3_conv3(x)

        x = self.p4_conv1(s8)
        x = self.p4_conv2(x)
        s16 = self.p4_conv3(x)

        x = self.p5_conv1(s16)
        x = self.p5_conv2(x)
        s32 = self.p5_conv3(x)

        p5 = self.latlayer1(s32)
        p4 = self._upsample_add(p5, self.latlayer2(s16))
        p3 = self._upsample_add(p4, self.latlayer3(s8))

        features.append(p3)
        features.append(p4)
        features.append(p5)

        for idx, t in enumerate(features):
            # print(idx)
            conf.append(self.cls_head[idx](t).permute(0, 2, 3, 1).contiguous())
            loc.append(self.box_head[idx](t).permute(0, 2, 3, 1).contiguous())
            # ldmk.append(self.ldmk_head[idx](t).permute(0, 2, 3, 1).contiguous())
            # attn.append(self.attn(t))

        # print(attn[0].size())
        # print(attn[1].size())
        # print(attn[2].size())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)


        features = torch.cat([o.view(o.size(0), o.size(1), -1) for o in features], -1).unsqueeze(-1)
        features = torch.cat([features, features], -1)
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)
        # print(features.size())
        # ldmk = torch.cat([o.view(o.size(0), -1) for o in ldmk], 1)

        if self.phase == "test":
            if self.num_classes == 2:
                output = (loc.view(loc.size(0), -1, 4),
                          self.softmax(conf.view(-1, self.num_classes))
                          )

            elif self.num_classes == 1:  # focal loss
                output = (loc.view(loc.size(0), -1, 4),
                          conf.view(-1, self.num_classes).sigmoid().max(1))

        else:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes))

        if self.out_featmaps:
            return output, features
        else:
            return output


class MDFace_imagenet(nn.Module):

    def __init__(self):
        super(MDFace_imagenet, self).__init__()
        self.conv1 = conv_bn(3, 8, 2)
        # self.conv1 = conv_bn_5X5(3, 12, 2)
        # self.conv2 = conv_bn(12, 12, 2)
        # self.conv1 = hetconv_bn_5X5(3, 12, num_groups=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.p3_conv1 = InvertedResidual(inp=8, oup=24, kernel=3, stride=2, expand_ratio=4)
        self.p3_conv2 = InvertedResidual(inp=24, oup=24, kernel=5, stride=1, expand_ratio=4)
        self.p3_conv3 = InvertedResidual(inp=24, oup=32, kernel=5, stride=1, expand_ratio=4)

        self.p4_conv1 = InvertedResidual(inp=32, oup=48, kernel=3, stride=2, expand_ratio=4)
        self.p4_conv2 = InvertedResidual(inp=48, oup=48, kernel=5, stride=1, expand_ratio=3)
        self.p4_conv3 = InvertedResidual(inp=48, oup=48, kernel=3, stride=1, expand_ratio=3)

        self.p5_conv1 = InvertedResidual(inp=48, oup=64, kernel=3, stride=2, expand_ratio=3)
        self.p5_conv2 = InvertedResidual(inp=64, oup=64, kernel=5, stride=1, expand_ratio=3)
        self.p5_conv3 = InvertedResidual(inp=64, oup=64, kernel=3, stride=1, expand_ratio=3)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(64, 1000)

    def forward(self, x):

        x = self.conv1(x)
        # s4 = self.conv2(x)
        s4 = self.maxpool1(x)

        x = self.p3_conv1(s4)
        x = self.p3_conv2(x)
        s8 = self.p3_conv3(x)

        x = self.p4_conv1(s8)
        x = self.p4_conv2(x)
        s16 = self.p4_conv3(x)

        x = self.p5_conv1(s16)
        x = self.p5_conv2(x)
        s32 = self.p5_conv3(x)

        x = self.avg_pool(s32)

        x = self.fc(x.view(x.size(0), -1))
        return x

if __name__=="__main__":
    import pdb
    pdb.set_trace()
    import sys
    sys.path.append(os.getcwd())
    from facedet.utils.misc import get_model_complexity_info

    input_size = (3, 224, 224)
    net = MDFace_imagenet()
    flops, params = get_model_complexity_info(net, input_size)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_size, flops, params))
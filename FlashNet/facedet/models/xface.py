# from . import backbones as backbones_mod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.operators
import math
import os
# from .backbones.mnasnet import _InvertedResidual, _BN_MOMENTUM
# from .backbones.se_module import SELayer
# from .cos_attn import Cos_Attn
# from .lrn_layer import LocalRelationalLayer

__all__ = ['XFace', 'XFace_dualpath']
class Scale(nn.Module):
    def __init__(self, init_value=2.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        #         print('self.scale',self.scale)
        return input * self.scale

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class XFace(nn.Module):

    def __init__(self, phase, cfg):
        super(XFace, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.out_featmaps = cfg['out_featmaps']
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

        self.p6_conv1 = InvertedResidual(inp=64, oup=80, kernel=3, stride=2, expand_ratio=3)
        self.p6_conv2 = InvertedResidual(inp=80, oup=80, kernel=5, stride=1, expand_ratio=3)
        self.p6_conv3 = InvertedResidual(inp=80, oup=80, kernel=3, stride=1, expand_ratio=3)
        # Lateral layers
        self.anchor_base_latlayer1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.anchor_base_latlayer2 = nn.Conv2d(48, 32, kernel_size=1, stride=1, padding=0)
        self.anchor_base_latlayer3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.anchor_free_latlayer1 = nn.Conv2d(80, 32, kernel_size=1, stride=1, padding=0)

        self.anchor_base_box_head, self.anchor_base_cls_head = self.multibox(self.num_classes)
        self.anchor_free_box_head = InvertedResidual_Head(inp=32, oup=1 * 4, kernel=3, stride=1, expand_ratio=1)
        self.anchor_free_cls_head = InvertedResidual_Head(inp=32, oup=1, kernel=3, stride=1, expand_ratio=1)
        self.anchor_free_ctr_head = InvertedResidual_Head(inp=32, oup=1, kernel=3, stride=1, expand_ratio=1)
        self.scale_op = nn.Sequential(*[Scale(init_value=1.0) for _ in range(1)])
        self.adp_maxpool = nn.AdaptiveMaxPool2d((10, 10))

        # self.attn = Cos_Attn()
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
        anchor_base_loc = list()
        anchor_base_conf = list()
        attentions = list()
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

        x = self.p6_conv1(s32)
        x = self.p6_conv2(x)
        s64 = self.p6_conv3(x)
        s64 = self.anchor_free_latlayer1(s64)

        _, _, s64_height, s64_width = s64.size()
        s64 = F.adaptive_max_pool2d(s64, (16, s64_width / (s64_height / 16)))

        p5 = self.anchor_base_latlayer1(s32)
        p4 = self._upsample_add(p5, self.anchor_base_latlayer2(s16))
        p3 = self._upsample_add(p4, self.anchor_base_latlayer3(s8))

        features.append(p3)
        features.append(p4)
        features.append(p5)

        attentions.append(p3.pow(2).mean(1).unsqueeze(1))
        attentions.append(p4.pow(2).mean(1).unsqueeze(1))
        attentions.append(p5.pow(2).mean(1).unsqueeze(1))
        attentions.append(self.anchor_free_cls_head(s64))

        # attentions.append(s8.pow(2).mean(1).unsqueeze(1))
        # attentions.append(s16.pow(2).mean(1).unsqueeze(1))
        # attentions.append(s32.pow(2).mean(1).unsqueeze(1))
        # attentions.append(s64.pow(2).mean(1).unsqueeze(1))

        for idx, t in enumerate(features):
            # print(idx)
            anchor_base_conf.append(self.anchor_base_cls_head[idx](t).permute(0, 2, 3, 1).contiguous())
            anchor_base_loc.append(self.anchor_base_box_head[idx](t).permute(0, 2, 3, 1).contiguous())

        anchor_base_loc = torch.cat([o.view(o.size(0), -1) for o in anchor_base_loc], 1)
        anchor_base_conf = torch.cat([o.view(o.size(0), -1) for o in anchor_base_conf], 1)

        anchor_free_loc = (self.scale_op(self.anchor_free_box_head(s64))).exp().permute(0, 2, 3, 1).contiguous()
        anchor_free_conf = self.anchor_free_cls_head(s64).permute(0, 2, 3, 1).contiguous()
        anchor_free_ctr = self.anchor_free_ctr_head(s64).permute(0, 2, 3, 1).contiguous()

        if self.phase == "test":
            output = (anchor_base_loc.view(anchor_base_loc.size(0), -1, 4),
                      self.softmax(anchor_base_conf.view(-1, 2)),
                      anchor_free_loc.view(anchor_base_conf.size(0), -1, 4),
                      anchor_free_conf.view(-1, 1).sigmoid(),
                      anchor_free_ctr.view(anchor_base_conf.size(0), -1, 1)
                      )
        # elif self.phase == "viz":

        else:
            output = (anchor_base_loc.view(anchor_base_loc.size(0), -1, 4),
                      anchor_base_conf.view(anchor_base_conf.size(0), -1, self.num_classes),
                      anchor_free_loc.view(anchor_base_conf.size(0), -1, 4),
                      anchor_free_conf.view(anchor_base_conf.size(0), -1, 1),
                      anchor_free_ctr.view(anchor_base_conf.size(0), -1, 1)
                      )

        if self.out_featmaps:
            return output, attentions
        else:
            return output


class XFace_dualpath(nn.Module):

    def __init__(self, phase, cfg):
        super(XFace_dualpath, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.out_featmaps = cfg['out_featmaps']
        self.finetune = cfg['finetune']

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
        self.anchor_base_latlayer1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.anchor_base_latlayer2 = nn.Conv2d(48, 32, kernel_size=1, stride=1, padding=0)
        self.anchor_base_latlayer3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.anchor_free_maxpool1 = nn.MaxPool2d(kernel_size=7, stride=4, padding=3)
        self.anchor_free_maxpool2 = nn.MaxPool2d(kernel_size=5, stride=4, padding=2)
        self.anchor_free_p5_conv1 = InvertedResidual(inp=32, oup=48, kernel=3, stride=2, expand_ratio=4)
        self.anchor_free_p5_conv2 = InvertedResidual(inp=48, oup=48, kernel=5, stride=1, expand_ratio=4)
        self.anchor_free_p6_conv1 = InvertedResidual(inp=48, oup=64, kernel=3, stride=2, expand_ratio=4)
        self.anchor_free_p6_conv2 = InvertedResidual(inp=64, oup=64, kernel=5, stride=1, expand_ratio=4)

        self.anchor_base_box_head, self.anchor_base_cls_head = self.anchor_base_head(self.num_classes)
        self.anchor_free_box_head, self.anchor_free_cls_head, self.anchor_free_ctr_head, self.anchor_free_scale_head = self.anchor_free_head()
        # self.adp_maxpool = nn.AdaptiveMaxPool2d((10, 10))

        # self.attn = Cos_Attn()
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def anchor_free_head(self):
        loc_layers = []
        conf_layers = []
        ctr_layers = []
        scale_layers = []
        channels = [48, 64]
        loc_layers += [InvertedResidual_Head(inp=channels[0], oup=1 * 4, kernel=3, stride=1, expand_ratio=1)]
        conf_layers += [InvertedResidual_Head(inp=channels[0], oup=1, kernel=3, stride=1, expand_ratio=1)]
        ctr_layers += [InvertedResidual_Head(inp=channels[0], oup=1, kernel=3, stride=1, expand_ratio=1)]
        scale_layers += [Scale(init_value=1.0)]

        loc_layers += [InvertedResidual_Head(inp=channels[1], oup=1 * 4, kernel=3, stride=1, expand_ratio=1)]
        conf_layers += [InvertedResidual_Head(inp=channels[1], oup=1, kernel=3, stride=1, expand_ratio=1)]
        ctr_layers += [InvertedResidual_Head(inp=channels[1], oup=1, kernel=3, stride=1, expand_ratio=1)]
        scale_layers += [Scale(init_value=1.0)]

        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers), nn.Sequential(*ctr_layers), nn.Sequential(*scale_layers)


    def anchor_base_head(self, num_classes):
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

        anchor_base_features = list()
        anchor_base_loc = list()
        anchor_base_conf = list()
        anchor_free_features = list()
        anchor_free_loc = list()
        anchor_free_conf = list()
        anchor_free_ctr = list()

        attentions = list()
        s2 = self.conv1(x)
        # s4 = self.conv2(x)
        s4 = self.maxpool1(s2)

        x = self.p3_conv1(s4)
        x = self.p3_conv2(x)
        s8 = self.p3_conv3(x)
        # print('out', s8.size())
        if not self.finetune:
            # print(s8.size())
            x = self.p4_conv1(s8)
            x = self.p4_conv2(x)
            s16 = self.p4_conv3(x)

            x = self.p5_conv1(s16)
            x = self.p5_conv2(x)
            s32 = self.p5_conv3(x)
            p5 = self.anchor_base_latlayer1(s32)
            p4 = self._upsample_add(p5, self.anchor_base_latlayer2(s16))
            p3 = self._upsample_add(p4, self.anchor_base_latlayer3(s8))
            anchor_base_features.append(p3)
            anchor_base_features.append(p4)
            anchor_base_features.append(p5)
            for idx, t in enumerate(anchor_base_features):
                # print(idx)
                anchor_base_conf.append(self.anchor_base_cls_head[idx](t).permute(0, 2, 3, 1).contiguous())
                anchor_base_loc.append(self.anchor_base_box_head[idx](t).permute(0, 2, 3, 1).contiguous())

            anchor_base_loc = torch.cat([o.view(o.size(0), -1) for o in anchor_base_loc], 1)
            anchor_base_conf = torch.cat([o.view(o.size(0), -1) for o in anchor_base_conf], 1)

        anchor_free_s32 = self.anchor_free_maxpool1(s8)

        x = self.anchor_free_p5_conv1(anchor_free_s32)
        anchor_free_s64 = self.anchor_free_p5_conv2(x)
        x = self.anchor_free_p6_conv1(anchor_free_s64)
        anchor_free_s128 = self.anchor_free_p6_conv2(x)
        # print(anchor_free_s64.size())
        # print(anchor_free_s128.size())
        anchor_free_features.append(anchor_free_s64)
        anchor_free_features.append(anchor_free_s128)

        # attentions.append(p3.pow(2).mean(1).unsqueeze(1))
        # attentions.append(p4.pow(2).mean(1).unsqueeze(1))
        # attentions.append(p5.pow(2).mean(1).unsqueeze(1))
        # attentions.append(self.anchor_free_cls_head(s32))

        for idx, t in enumerate(anchor_free_features):
            # print(t.size())
            anchor_free_loc.append((self.anchor_free_scale_head[idx](self.anchor_free_box_head[idx](t))).exp().permute(0, 2, 3, 1).contiguous())
            anchor_free_conf.append(self.anchor_free_cls_head[idx](t).permute(0, 2, 3, 1).contiguous())
            anchor_free_ctr.append(self.anchor_free_ctr_head[idx](t).permute(0, 2, 3, 1).contiguous())

        anchor_free_loc = torch.cat([o.view(o.size(0), -1) for o in anchor_free_loc], 1)
        anchor_free_conf = torch.cat([o.view(o.size(0), -1) for o in anchor_free_conf], 1)
        anchor_free_ctr = torch.cat([o.view(o.size(0), -1) for o in anchor_free_ctr], 1)


        if self.phase == "test":
            output = (anchor_base_loc.view(anchor_base_loc.size(0), -1, 4),
                      self.softmax(anchor_base_conf.view(-1, 2)),

                      anchor_free_loc.view(anchor_base_conf.size(0), -1, 4),
                      anchor_free_conf.view(-1, 1).sigmoid(),
                      anchor_free_ctr.view(anchor_base_conf.size(0), -1, 1)
                      )
        # elif self.phase == "viz":
        else:
            if not self.finetune:
                output = (anchor_base_loc.view(anchor_base_loc.size(0), -1, 4),
                          anchor_base_conf.view(anchor_base_conf.size(0), -1, self.num_classes),

                          anchor_free_loc.view(anchor_free_conf.size(0), -1, 4),
                          anchor_free_conf.view(anchor_free_conf.size(0), -1, 1),
                          anchor_free_ctr.view(anchor_free_conf.size(0), -1, 1)
                          )
            else:
                output = (None,
                          None,
                          anchor_free_loc.view(anchor_free_conf.size(0), -1, 4),
                          anchor_free_conf.view(anchor_free_conf.size(0), -1, 1),
                          anchor_free_ctr.view(anchor_free_conf.size(0), -1, 1)
                          )

        if self.out_featmaps:
            return output, attentions
        else:
            return output


class XFace_AdpPool(nn.Module):

    def __init__(self, phase, cfg):
        super(XFace_AdpPool, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.out_featmaps = cfg['out_featmaps']
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

        self.p6_conv1 = InvertedResidual(inp=64, oup=80, kernel=3, stride=2, expand_ratio=3)
        self.p6_conv2 = InvertedResidual(inp=80, oup=80, kernel=5, stride=1, expand_ratio=3)
        self.p6_conv3 = InvertedResidual(inp=80, oup=80, kernel=3, stride=1, expand_ratio=3)
        # Lateral layers
        self.anchor_base_latlayer1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.anchor_base_latlayer2 = nn.Conv2d(48, 32, kernel_size=1, stride=1, padding=0)
        self.anchor_base_latlayer3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.anchor_free_latlayer1 = nn.Conv2d(80, 32, kernel_size=1, stride=1, padding=0)

        self.anchor_base_box_head, self.anchor_base_cls_head = self.multibox(self.num_classes)
        self.anchor_free_box_head = InvertedResidual_Head(inp=32, oup=1 * 4, kernel=3, stride=1, expand_ratio=1)
        self.anchor_free_cls_head = InvertedResidual_Head(inp=32, oup=1, kernel=3, stride=1, expand_ratio=1)
        self.anchor_free_ctr_head = InvertedResidual_Head(inp=32, oup=1, kernel=3, stride=1, expand_ratio=1)
        self.scale_op = nn.Sequential(*[Scale(init_value=1.0) for _ in range(1)])
        self.adp_maxpool = nn.AdaptiveMaxPool2d((10, 10))

        # self.attn = Cos_Attn()
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
        anchor_base_loc = list()
        anchor_base_conf = list()
        attentions = list()
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

        x = self.p6_conv1(s32)
        x = self.p6_conv2(x)
        s64 = self.p6_conv3(x)
        s64 = self.anchor_free_latlayer1(s64)

        p5 = self.anchor_base_latlayer1(s32)
        p4 = self._upsample_add(p5, self.anchor_base_latlayer2(s16))
        p3 = self._upsample_add(p4, self.anchor_base_latlayer3(s8))

        features.append(p3)
        features.append(p4)
        features.append(p5)

        attentions.append(p3.pow(2).mean(1).unsqueeze(1))
        attentions.append(p4.pow(2).mean(1).unsqueeze(1))
        attentions.append(p5.pow(2).mean(1).unsqueeze(1))
        attentions.append(self.anchor_free_cls_head(s64))

        # attentions.append(s8.pow(2).mean(1).unsqueeze(1))
        # attentions.append(s16.pow(2).mean(1).unsqueeze(1))
        # attentions.append(s32.pow(2).mean(1).unsqueeze(1))
        # attentions.append(s64.pow(2).mean(1).unsqueeze(1))

        for idx, t in enumerate(features):
            # print(idx)
            anchor_base_conf.append(self.anchor_base_cls_head[idx](t).permute(0, 2, 3, 1).contiguous())
            anchor_base_loc.append(self.anchor_base_box_head[idx](t).permute(0, 2, 3, 1).contiguous())

        anchor_base_loc = torch.cat([o.view(o.size(0), -1) for o in anchor_base_loc], 1)
        anchor_base_conf = torch.cat([o.view(o.size(0), -1) for o in anchor_base_conf], 1)

        anchor_free_loc = (self.scale_op(self.anchor_free_box_head(s64))).exp().permute(0, 2, 3, 1).contiguous()
        anchor_free_conf = self.anchor_free_cls_head(s64).permute(0, 2, 3, 1).contiguous()
        anchor_free_ctr = self.anchor_free_ctr_head(s64).permute(0, 2, 3, 1).contiguous()

        if self.phase == "test":
            output = (anchor_base_loc.view(anchor_base_loc.size(0), -1, 4),
                      self.softmax(anchor_base_conf.view(-1, 2)),
                      anchor_free_loc.view(anchor_base_conf.size(0), -1, 4),
                      anchor_free_conf.view(-1, 1).sigmoid(),
                      anchor_free_ctr.view(anchor_base_conf.size(0), -1, 1)
                      )
        # elif self.phase == "viz":

        else:
            output = (anchor_base_loc.view(anchor_base_loc.size(0), -1, 4),
                      anchor_base_conf.view(anchor_base_conf.size(0), -1, self.num_classes),
                      anchor_free_loc.view(anchor_base_conf.size(0), -1, 4),
                      anchor_free_conf.view(anchor_base_conf.size(0), -1, 1),
                      anchor_free_ctr.view(anchor_base_conf.size(0), -1, 1)
                      )

        if self.out_featmaps:
            return output, attentions
        else:
            return output


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_bn_5X5(inp, oup, stride):

    return nn.Sequential(
        nn.Conv2d(inp, oup, 5, stride, 2, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        if kernel == 3:
            padding = 1
        else:
            padding = 2
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, padding, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, padding, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual_Head(nn.Module):
    def __init__(self, inp, oup, kernel, stride, expand_ratio):
        super(InvertedResidual_Head, self).__init__()
        self.stride = stride
        if kernel == 3:
            padding = 1
        else:
            padding = 2
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, padding, groups=hidden_dim, bias=False),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, padding, groups=hidden_dim, bias=False),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),

            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MDFace_imagenet(nn.Module):

    def __init__(self):
        super(MDFace_imagenet, self).__init__()
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
    from facedet.utils.misc import get_model_complexity_info
    input_size = (3, 224, 224)
    net = MDFace_imagenet()
    flops, params = get_model_complexity_info(net, input_size)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_size, flops, params))

    net = net.cuda()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        out = net(torch.rand(1, 3, 640, 640).cuda())
        print(prof)
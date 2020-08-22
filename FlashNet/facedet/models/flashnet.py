import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.operators
import math
import os
from ...facedet.models.common_ops import *

__all__ = ['FlashNet']
class FlashNet(nn.Module):

    def __init__(self, phase, cfg):
        super(FlashNet, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.out_featmaps = cfg['out_featmaps']
        self.num_anchors_per_featmap = cfg['num_anchors_per_featmap']
        self.feat_adp = cfg['feat_adp']
        self.use_ldmk = cfg['use_ldmk']
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

        self.box_head, self.cls_head, self.ldmk_head = self.multibox(self.num_classes)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []
        ldmk_layers = []
        channels = [32, 32, 32]
        loc_layers += [InvertedResidual_Head(inp=channels[0], oup=self.num_anchors_per_featmap * 4, kernel=3, stride=1, expand_ratio=1)]
        conf_layers += [InvertedResidual_Head(inp=channels[0], oup=self.num_anchors_per_featmap * num_classes, kernel=3, stride=1, expand_ratio=1)]
        ldmk_layers += [InvertedResidual_Head(inp=channels[0], oup=self.num_anchors_per_featmap * 10, kernel=3, stride=1, expand_ratio=1)]
        loc_layers += [InvertedResidual_Head(inp=channels[1], oup=self.num_anchors_per_featmap * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[1], oup=self.num_anchors_per_featmap * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        ldmk_layers += [InvertedResidual_Head(inp=channels[0], oup=self.num_anchors_per_featmap * 10, kernel=3, stride=1, expand_ratio=0.5)]
        loc_layers += [InvertedResidual_Head(inp=channels[2], oup=self.num_anchors_per_featmap * 4, kernel=3, stride=1, expand_ratio=0.5)]
        conf_layers += [InvertedResidual_Head(inp=channels[2], oup=self.num_anchors_per_featmap * num_classes, kernel=3, stride=1, expand_ratio=0.5)]
        ldmk_layers += [InvertedResidual_Head(inp=channels[0], oup=self.num_anchors_per_featmap * 10, kernel=3, stride=1, expand_ratio=0.5)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers), nn.Sequential(*ldmk_layers)

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
            conf.append(self.cls_head[idx](t).permute(0, 2, 3, 1).contiguous())
            loc.append(self.box_head[idx](t).permute(0, 2, 3, 1).contiguous())
            if self.use_ldmk:
                ldmk.append(self.ldmk_head[idx](t).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.use_ldmk:
            ldmk = torch.cat([o.view(o.size(0), -1) for o in ldmk], 1)

        if self.phase == "test":
            conf = self.softmax(conf.view(conf.size(0), -1, self.num_classes))
        else:
            conf = conf.view(conf.size(0), -1, self.num_classes)

        if self.use_ldmk:
            output = (ldmk.view(loc.size(0), -1, 10),
                        loc.view(loc.size(0), -1, 4),
                        conf,
                        )
        else:
            output = (loc.view(loc.size(0), -1, 4),
                        conf
                    )
            # elif self.num_classes == 1:  # focal loss
            #     output = (loc.view(loc.size(0), -1, 4),
            #               conf.view(-1, self.num_classes).sigmoid().max(1))
        return output
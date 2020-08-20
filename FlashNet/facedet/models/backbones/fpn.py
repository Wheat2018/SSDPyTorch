import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet as vrn

from .resnet import ResNet
from .mnasnet import MNASNet, MNASNet_S6
from .utils import register
from .se_module import SELayer

class FPN(nn.Module):
    'Feature Pyramid Network - https://arxiv.org/abs/1612.03144'

    def __init__(self, features):
        super().__init__()

        self.stride = 128
        self.features = features

        is_light = features.bottleneck == vrn.BasicBlock
        channels = [128, 256, 512] if is_light else [512, 1024, 2048]

        self.lateral3 = nn.Conv2d(channels[0], 256, 1)
        self.lateral4 = nn.Conv2d(channels[1], 256, 1)
        self.lateral5 = nn.Conv2d(channels[2], 256, 1)
        self.pyramid6 = nn.Conv2d(channels[2], 256, 3, stride=2, padding=1)
        self.pyramid7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        self.apply(init_layer)

        self.features.initialize()

    def forward(self, x):
        c3, c4, c5 = self.features(x)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        # p4 = F.interpolate(p5, scale_factor=2) + p4
        p3 = F.interpolate(p4, size=[p4.size(2), p4.size(3)]) + p4
        p3 = self.lateral3(c3)
        # print(p3.size())
        # print(F.interpolate(p4, scale_factor=2).size())
        p3 = F.interpolate(p4, size=[p3.size(2), p3.size(3)]) + p3
        # p3 = F.interpolate(p4, scale_factor=2) + p3

        p6 = self.pyramid6(c5)
        p7 = self.pyramid7(F.relu(p6))

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)
        # print(p3.size())
        # print(p4.size())
        # print(p5.size())
        # print(p6.size())
        # print(p7.size())

        return [p3, p4, p5, p6, p7]


class MNASNet_FPN(nn.Module):
    'Feature Pyramid Network - https://arxiv.org/abs/1612.03144'

    def __init__(self, features):
        super().__init__()

        self.stride = 128
        self.features = features
        channels = [16, 24, 40, 48, 96]

        self.lateral3 = nn.Conv2d(channels[0], 48, 1)
        self.lateral4 = nn.Conv2d(channels[1], 48, 1)
        self.lateral5 = nn.Conv2d(channels[2], 48, 1)
        self.lateral6 = nn.Conv2d(channels[3], 48, 1)
        self.lateral7 = nn.Conv2d(channels[4], 48, 1)

        self.smooth3 = nn.Conv2d(48, 48, 3, padding=1)
        self.smooth4 = nn.Conv2d(48, 48, 3, padding=1)
        self.smooth5 = nn.Conv2d(48, 48, 3, padding=1)
        self.smooth6 = nn.Conv2d(48, 48, 3, padding=1)
        self.smooth7 = nn.Conv2d(48, 48, 3, padding=1)

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        self.apply(init_layer)

        self.features.initialize()

    def forward(self, x):
        c3, c4, c5, c6, c7 = self.features(x)

        p7 = self.lateral7(c7)
        p6 = self.lateral6(c6)
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3)

        p6 = F.interpolate(p7, size=[p6.size(2), p6.size(3)]) + p6
        p5 = F.interpolate(p6, size=[p5.size(2), p5.size(3)]) + p5
        p4 = F.interpolate(p5, size=[p4.size(2), p4.size(3)]) + p4
        p3 = F.interpolate(p4, size=[p3.size(2), p3.size(3)]) + p3

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)
        p6 = self.smooth4(p6)
        p7 = self.smooth5(p7)
        # print(p3.size())
        # print(p4.size())
        # print(p5.size())
        # print(p6.size())
        # print(p7.size())

        return [p3, p4, p5, p6, p7]

class MNASNet_FPN_SE(nn.Module):
    'Feature Pyramid Network - https://arxiv.org/abs/1612.03144'

    def __init__(self, features):
        super().__init__()

        self.stride = 128
        self.features = features
        self.lateral_channel = 32
        channels = [32, 40, 48, 56, 64]

        self.lateral3 = nn.Conv2d(channels[0], self.lateral_channel, 1)
        self.lateral4 = nn.Conv2d(channels[1], self.lateral_channel, 1)
        self.lateral5 = nn.Conv2d(channels[2], self.lateral_channel, 1)
        self.lateral6 = nn.Conv2d(channels[3], self.lateral_channel, 1)
        self.lateral7 = nn.Conv2d(channels[4], self.lateral_channel, 1)

        self.smooth3 = nn.Conv2d(self.lateral_channel, self.lateral_channel, 3, groups=self.lateral_channel, padding=1)
        self.smooth4 = nn.Conv2d(self.lateral_channel, self.lateral_channel, 3, groups=self.lateral_channel, padding=1)
        self.smooth5 = nn.Conv2d(self.lateral_channel, self.lateral_channel, 3, groups=self.lateral_channel, padding=1)
        self.smooth6 = nn.Conv2d(self.lateral_channel, self.lateral_channel, 3, groups=self.lateral_channel, padding=1)
        self.smooth7 = nn.Conv2d(self.lateral_channel, self.lateral_channel, 3, groups=self.lateral_channel, padding=1)

        self.p3_se = SELayer(channel=self.lateral_channel, reduction=8)
        self.p4_se = SELayer(channel=self.lateral_channel, reduction=8)
        self.p5_se = SELayer(channel=self.lateral_channel, reduction=8)
        self.p6_se = SELayer(channel=self.lateral_channel, reduction=8)
        self.p7_se = SELayer(channel=self.lateral_channel, reduction=8)

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        self.apply(init_layer)

        self.features.initialize()

    def forward(self, x):
        c3, c4, c5, c6, c7 = self.features(x)

        p7 = self.lateral7(c7)
        p6 = self.lateral6(c6)
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3)

        p6 = F.interpolate(p7, size=[p6.size(2), p6.size(3)]) + p6
        p5 = F.interpolate(p6, size=[p5.size(2), p5.size(3)]) + p5
        p4 = F.interpolate(p5, size=[p4.size(2), p4.size(3)]) + p4
        p3 = F.interpolate(p4, size=[p3.size(2), p3.size(3)]) + p3

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)
        p6 = self.smooth4(p6)
        p7 = self.smooth5(p7)

        # p3 = self.p3_se(p3)
        # p4 = self.p4_se(p4)
        # p5 = self.p5_se(p5)
        # p6 = self.p6_se(p6)
        # p7 = self.p7_se(p7)

        # print(p3.size())
        # print(p4.size())
        # print(p5.size())
        # print(p6.size())
        # print(p7.size())

        return [p3, p4, p5, p6, p7]

class MNASNet_FPN_S6_SE(nn.Module):
    'Feature Pyramid Network - https://arxiv.org/abs/1612.03144'

    def __init__(self, features):
        super().__init__()

        self.stride = 128
        self.features = features
        self.lateral_channel = 32
        channels = [32, 40, 48, 56, 64, 72]

        self.lateral3 = nn.Conv2d(channels[0], self.lateral_channel, 1)
        self.lateral4 = nn.Conv2d(channels[1], self.lateral_channel, 1)
        self.lateral5 = nn.Conv2d(channels[2], self.lateral_channel, 1)
        self.lateral6 = nn.Conv2d(channels[3], self.lateral_channel, 1)
        self.lateral7 = nn.Conv2d(channels[4], self.lateral_channel, 1)
        self.lateral8 = nn.Conv2d(channels[5], self.lateral_channel, 1)

        self.smooth3 = nn.Conv2d(self.lateral_channel, self.lateral_channel, 3, groups=self.lateral_channel, padding=1)
        self.smooth4 = nn.Conv2d(self.lateral_channel, self.lateral_channel, 3, groups=self.lateral_channel, padding=1)
        self.smooth5 = nn.Conv2d(self.lateral_channel, self.lateral_channel, 3, groups=self.lateral_channel, padding=1)
        self.smooth6 = nn.Conv2d(self.lateral_channel, self.lateral_channel, 3, groups=self.lateral_channel, padding=1)
        self.smooth7 = nn.Conv2d(self.lateral_channel, self.lateral_channel, 3, groups=self.lateral_channel, padding=1)
        self.smooth8 = nn.Conv2d(self.lateral_channel, self.lateral_channel, 3, groups=self.lateral_channel, padding=1)

        self.p3_se = SELayer(channel=self.lateral_channel, reduction=8)
        self.p4_se = SELayer(channel=self.lateral_channel, reduction=8)
        self.p5_se = SELayer(channel=self.lateral_channel, reduction=8)
        self.p6_se = SELayer(channel=self.lateral_channel, reduction=8)
        self.p7_se = SELayer(channel=self.lateral_channel, reduction=8)
        self.p8_se = SELayer(channel=self.lateral_channel, reduction=8)

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        self.apply(init_layer)

        self.features.initialize()

    def forward(self, x):
        c3, c4, c5, c6, c7, c8 = self.features(x)

        p8 = self.lateral8(c8)
        p7 = self.lateral7(c7)
        p6 = self.lateral6(c6)
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3)

        p7 = F.interpolate(p8, size=[p7.size(2), p7.size(3)]) + p7
        p6 = F.interpolate(p7, size=[p6.size(2), p6.size(3)]) + p6
        p5 = F.interpolate(p6, size=[p5.size(2), p5.size(3)]) + p5
        p4 = F.interpolate(p5, size=[p4.size(2), p4.size(3)]) + p4
        p3 = F.interpolate(p4, size=[p3.size(2), p3.size(3)]) + p3

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)
        p6 = self.smooth4(p6)
        p7 = self.smooth5(p7)
        p8 = self.smooth6(p8)

        # p3 = self.p3_se(p3)
        # p4 = self.p4_se(p4)
        # p5 = self.p5_se(p5)
        # p6 = self.p6_se(p6)
        # p7 = self.p7_se(p7)

        # print(p3.size())
        # print(p4.size())
        # print(p5.size())
        # print(p6.size())
        # print(p7.size())

        return [p3, p4, p5, p6, p7, p8]

@register
def ResNet18FPN():
    return FPN(ResNet(layers=[2, 2, 2, 2], bottleneck=vrn.BasicBlock, outputs=[3, 4, 5], url=vrn.model_urls['resnet18']))

@register
def ResNet34FPN():
    return FPN(ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.BasicBlock, outputs=[3, 4, 5], url=vrn.model_urls['resnet34']))

@register
def ResNet50FPN():
    return FPN(ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, outputs=[3, 4, 5], url=vrn.model_urls['resnet50']))

@register
def ResNet101FPN():
    return FPN(ResNet(layers=[3, 4, 23, 3], bottleneck=vrn.Bottleneck, outputs=[3, 4, 5], url=vrn.model_urls['resnet101']))

@register
def ResNet152FPN():
    return FPN(ResNet(layers=[3, 8, 36, 3], bottleneck=vrn.Bottleneck, outputs=[3, 4, 5], url=vrn.model_urls['resnet152']))

@register
def MNASNet0_5_FPN():
    return MNASNet_FPN(MNASNet(alpha=0.5))

@register
def MNASNet1_0_FPN_SE():
    return MNASNet_FPN_SE(MNASNet(alpha=1))


@register
def MNASNet_FPN_S6_SE_func():
    return MNASNet_FPN_S6_SE(MNASNet_S6(alpha=1))
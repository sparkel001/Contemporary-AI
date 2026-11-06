import torch
from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.ops import SqueezeExcitation


class SEBlock(BasicBlock):
    def __init__(self, inplanes, planes, *args, **kwargs):
        super().__init__(inplanes, planes, *args, **kwargs)
        self.bn2 = nn.Sequential(self.bn2, SqueezeExcitation(planes, max(1, planes // 16)))


class SmallResNet(ResNet):
    """Four two-block stages with a CIFAR stem and reduced width."""
    def __init__(self, block):
        nn.Module.__init__(self)
        self.inplanes = 16
        self._norm_layer = nn.BatchNorm2d
        self.groups = 1
        self.base_width = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()
        for stage, width in enumerate([16, 32, 64, 128], 1):
            setattr(self, f'layer{stage}', self._make_layer(block, width, 2, stride=1 if stage == 1 else 2))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)


def model(name):
    if name in ('resnet', 'se'):
        return SmallResNet(SEBlock if name == 'se' else BasicBlock)
    if name != 'cnn':
        raise ValueError(name)
    layers = []
    channels = 3
    for width in (32, 64, 128):
        for _ in range(2):
            layers.extend([nn.Conv2d(channels, width, 3, padding=1, bias=False),
                           nn.BatchNorm2d(width), nn.ReLU(inplace=True)])
            channels = width
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))

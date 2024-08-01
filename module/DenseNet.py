import torch
import torch.nn as nn
from collections import OrderedDict

import torchvision.models
from torchvision import models

import torch.nn.functional as F


class _DenseLayer(nn.Sequential):
    """DenseNet中的卷积模块"""

    def __init__(self, in_features_num, bn_size, growth_rate, drop_rate):
        """
        Args:
            in_features_num: 输入模块的特征图大小
            bn_size: 用来控制特征图通道数量增长的
            growth_rate: 经过DenseBlock模块后输出的特征图通道数量
            drop_rate: 神经元失活率, 用来提高模型泛化能力
        """
        super(_DenseLayer, self).__init__()

        self.in_features_num = in_features_num
        self.bn_size = bn_size
        self.k = growth_rate

        self.drop_rate = drop_rate

        self.dense_layer = nn.Sequential(
            OrderedDict(
                # 第一个卷积核模块
                [('norm1', nn.BatchNorm2d(self.in_features_num)),
                 ('relu1', nn.ReLU(inplace=True)),
                 ('conv1', nn.Conv2d(in_channels=self.in_features_num, out_channels=self.bn_size * self.k,
                                     kernel_size=1, stride=1, bias=False)),
                 # 第二个卷积核模块
                 ('norm2', nn.BatchNorm2d(self.bn_size * self.k)),
                 ('relu2', nn.ReLU(inplace=True)),
                 ('conv2', nn.Conv2d(in_channels=self.bn_size * self.k, out_channels=self.k,
                                     kernel_size=3, stride=1, padding=1, bias=False))
                 ]
            )
        )

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_features_num, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_features_num + i * growth_rate, bn_size, growth_rate, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, in_features_num, out_features_num):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_features_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_features_num, out_features_num, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0):
        super(DenseNet121, self).__init__()

        # 初始卷积层
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # DenseBlock 和 Transition 层
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, in_features_num=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(in_features_num=num_features, out_features_num=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # 最后的批量归一化层
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # # 分类器
        # self.classifier = nn.Linear(num_features, 1000)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        # out = self.classifier(out)
        return out


# densenet121 = DenseNet121()

# encoder = DenseNet121()
# print(encoder(torch.randn(1, 3, 224, 224)).shape)  # torch.Size([1, 1024])


def densenet121():
    """

    :return: densenet121
    """
    return DenseNet121(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0)


# print(densenet121())
# for c in list(densenet121().children())[5:]:
#     for p in c.parameters():
#         print(p)
# print("_" * 100)
# print(torchvision.models.densenet121())

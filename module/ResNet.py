import torch
import torch.nn.functional as F
import torch.nn as nn


# 这里实现一个创建模块(conv+bn)的方法(ResNet中频繁的的使用)
import torchvision.models
from torchvision.transforms import transforms


def make_conv_bn(conv_type, in_planes, out_planes, kernel_size, stride, padding, bias):
    return nn.Sequential(
        conv_type(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_planes)
    )


class BasicBlock(nn.Module):
    """
    这里是一个resnet中的模块
    """

    # 每个残差块(Residual Block)的输出通道数通常是输入通道数的某个倍数, 这在resnet50以后的网络会用很多, 18可以忽略
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, conv_type=None):
        super(BasicBlock, self).__init__()

        if conv_type is None:
            conv_type = nn.Conv2d

        # 模块第一个卷积
        self.conv1 = make_conv_bn(conv_type, in_planes, out_planes,
                                  kernel_size=3, stride=stride, padding=1, bias=False)
        # 模块第二个卷积
        self.conv2 = make_conv_bn(conv_type, out_planes, out_planes,
                                  kernel_size=3, stride=1, padding=1, bias=False)

        # 用于存储残差连接(residual connection)
        self.shortcut = nn.Sequential()
        # 这里处理残差连结的部分
        if stride != 1 or in_planes != out_planes * self.expansion:
            # 当不同的BasicBlock块进行残差连结的的时候
            # 直接将两个特征图相加相加是不可能的, 因为它们的尺寸不同, 无法完成跨层连接
            # 需要引入一个额外的卷积层来调整输入特征图的尺寸
            self.shortcut = nn.Sequential(
                # 使用1x1卷积层改变通道数
                conv_type(in_planes, out_planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * self.expansion)
            )

    def forward(self, x):
        # 在这里拼接模块个个模块
        out = F.relu(self.conv1(x))
        out = self.conv2(out) + self.shortcut(x)
        out = F.relu(out)

        return out


# 这是构建res50使用的一个模块类与BasicBlock类似
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, conv_type=None):

        super(Bottleneck, self).__init__()

        if conv_type is None:
            conv_type = nn.Conv2d

        # 第一个卷积层
        self.conv1 = conv_type(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # 第二个卷积层
        self.conv2 = conv_type(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # 第三个卷积层
        self.conv3 = conv_type(out_planes, out_planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * 4)

        self.relu = nn.ReLU(inplace=True)

        # shortcut连接, 用于处理特征图维度不匹配的情况, 做法同上
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes * 4:
            self.shortcut = nn.Sequential(
                conv_type(in_planes, out_planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * 4)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(x)  # 将shortcut连接的结果与主路径的输出相加

        out = self.relu(out)  # 使用ReLU激活函数
        return out


class ResNet18(nn.Module):

    def __init__(self, block, layers, num_classes=1000, conv_type=None):
        super(ResNet18, self).__init__()

        if conv_type is None:
            conv_type = nn.Conv2d

        self.in_planes = 64  # resnet第一层卷积输出图的大小是固定的

        # 这是第一层(resnet18)卷积
        self.conv0 = make_conv_bn(conv_type=conv_type, in_planes=3, out_planes=self.in_planes, kernel_size=7,
                                  stride=2, padding=3, bias=False)

        # 最大池化后进入BasicBlock模块
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个模块卷积输出图特征的大小依次是[64, 128, 256, 512]
        self.layer1 = self._make_seq_layer(block, 64, layers[0], stride=1, conv_type=conv_type)
        # 这里之后的步长是2
        self.layer2 = self._make_seq_layer(block, 128, num_blocks=layers[1], stride=2, conv_type=conv_type)
        self.layer3 = self._make_seq_layer(block, 256, num_blocks=layers[2], stride=2, conv_type=conv_type)
        self.layer4 = self._make_seq_layer(block, 512, num_blocks=layers[3], stride=2, conv_type=conv_type)

        # 特征图输出部分的处理
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # 定义一个残差网络序列
    def _make_seq_layer(self, block, out_planes, num_blocks, stride, conv_type):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(in_planes=self.in_planes, out_planes=out_planes, stride=stride, conv_type=conv_type))
            self.in_planes = out_planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        x = F.relu(self.conv0(x))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(conv_type=None):
    # resnet中每个BasicBlock块都是两层
    return ResNet18(BasicBlock, [2, 2, 2, 2], conv_type=conv_type)


class ResNet50(nn.Module):
    """
    ResNet50模型的定义, 包括修改第一层卷积、BasicBlock模块以及特征图输出部分的处理

    Args:
        block: ResNet中的基本残差块类型, 如Bottleneck
        layers: 各阶段的残差块重复次数, 如[3, 4, 6, 3]
        num_classes: 最终输出的类别数量, 默认为1000
        conv_type: 卷积层类型，默认为None, 即nn.Conv2d

    Attributes:
        in_planes: 当前输入特征图的通道数, 初始化为64

    """

    def __init__(self, block, layers, num_classes=1000, conv_type=None):
        super(ResNet50, self).__init__()

        if conv_type is None:
            conv_type = nn.Conv2d

        self.in_planes = 64  # ResNet50第一层卷积输出图的大小是64

        # 第一层卷积
        self.conv0 = make_conv_bn(conv_type=conv_type, in_planes=3, out_planes=self.in_planes, kernel_size=7,
                                  stride=2, padding=3, bias=False)

        # 最大池化后进入BasicBlock模块
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet50的结构和18类似
        self.layer1 = self._make_seq_layer(block, 64, layers[0], stride=1, conv_type=conv_type)
        self.layer2 = self._make_seq_layer(block, 128, num_blocks=layers[1], stride=2, conv_type=conv_type)
        self.layer3 = self._make_seq_layer(block, 256, num_blocks=layers[2], stride=2, conv_type=conv_type)
        self.layer4 = self._make_seq_layer(block, 512, num_blocks=layers[3], stride=2, conv_type=conv_type)

        # 特征图输出部分的处理
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # 连结个个模块
    def _make_seq_layer(self, block, out_planes, num_blocks, stride, conv_type):

        """
        构建一个阶段的残差块序列

        Args:
            block: ResNet中的基本残差块类型, 如Bottleneck
            out_planes: 输出特征图的通道数
            num_blocks: 当前阶段的残差块重复次数
            stride: 步长
            conv_type: 卷积层类型

        Returns:
            nn.Sequential: 包含多个残差块的序列

        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(in_planes=self.in_planes, out_planes=out_planes, stride=stride, conv_type=conv_type))
            self.in_planes = out_planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        """
        前向传播过程, 包括卷积, 池化, 残差块序列, 全局平均池化和全连接层。

        Args:
            x: 输入的特征图

        Returns:
            torch.Tensor: 模型输出

        """

        x = F.relu(self.conv0(x))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(conv_type=None):
    """
    创建一个ResNet50模型实例

    Args:
        conv_type: 卷积层类型，默认为None

    Returns:
        ResNet50: ResNet50模型实例

    """
    return ResNet50(Bottleneck, [3, 4, 6, 3], conv_type=conv_type)


# print(torchvision.models.resnet50(pretrained=True))

# print("_"*100)
# r = resnet50()
# print(r)
#
# standard = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  # 调整图像大小
#     transforms.ToTensor(),  # 转换回张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
# ])
#
# batch_size = 2
# image_size = 224  # 输入图像尺寸为224x224
# images_x = torch.randn(batch_size, 3, image_size, image_size)
#
# encoder = resnet18()
#
# encoded_images = encoder(images_x)
#
# # 输出编码结果的尺寸
# print(f'images shape: {encoded_images.shape}')

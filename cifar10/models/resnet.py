import torch.nn as nn

class ResNet20(nn.Module):
    """
    后面有bn bias可以是Fasle
    nn.AdaptiveAvgPool2d好用
    nn.Sequential好用
    """

    def __init__(self, in_channel=3, num_classes=10):
        super(ResNet20, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(2, 2)
        self.block1 = self._make_layer(16, 16, 3)
        self.block2 = self._make_layer(16, 32, 3, downsample=True)
        self.block3 = self._make_layer(32, 64, 3, downsample=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channel, out_channel, num_block, downsample=False):
        layers = []
        if downsample:
            downsample = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False),
                                       nn.BatchNorm2d(out_channel))
        else:
            downsample = None

        layers.append(ResNetBlock(in_channel, out_channel, downsample))
        for i in range(1, num_block):
            layers.append(ResNetBlock(out_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.view((x.shape[0], -1))
        x = self.fc(x)
        return x


class ResNet18(nn.Module):
    """
    后面有bn bias可以是Fasle
    nn.AdaptiveAvgPool2d好用
    nn.Sequential好用
    """

    def __init__(self, in_channel, num_classes=1000):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.block1 = self._make_layer(64, 64, 3)
        self.block2 = self._make_layer(64, 128, 3, downsample=True)
        self.block3 = self._make_layer(128, 256, 6, downsample=True)
        self.block4 = self._make_layer(256, 512, 3, downsample=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channel, out_channel, num_block, downsample=False):
        layers = []
        if downsample:
            downsample = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False),
                                       nn.BatchNorm2d(out_channel))
        else:
            downsample = None

        layers.append(ResNetBlock(in_channel, out_channel, downsample))
        for i in range(1, num_block):
            layers.append(ResNetBlock(out_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.view((x.shape[0], -1))
        x = self.fc(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=None):
        super(ResNetBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.downsample = downsample
        if downsample is None:
            self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                   stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out = out + x  # TODO together batch normalization??
        out = self.relu(out)
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

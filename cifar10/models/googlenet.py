import torch
import torch.nn as nn
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.parameter_count = 0

        # 在ImageNet上分类的第一个卷积核是7×7stide=2的
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(192)

        self.inc3a = self._make_inception(192, 64, 96, 128, 16, 32, 32, 256)
        self.inc3b = self._make_inception(256, 128, 128, 192, 32, 96, 64, 480)

        self.inc4a = self._make_inception(480, 192, 96, 208, 16, 48, 64, 512)
        self.inc4b = self._make_inception(512, 160, 112, 224, 24, 64, 64, 512)
        self.inc4c = self._make_inception(512, 128, 128, 256, 24, 64, 64, 512)
        self.inc4d = self._make_inception(512, 112, 144, 288, 32, 64, 64, 528)
        self.inc4e = self._make_inception(528, 256, 160, 320, 32, 128, 128, 832)

        self.inc5a = self._make_inception(832, 256, 160, 320, 32, 128, 128, 832)
        self.inc5b = self._make_inception(832, 384, 192, 384, 48, 128, 128, 1024)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2, ceil_mode=True)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2, ceil_mode=True)

        x = self._inception_forward(x, self.inc3a)
        x = self._inception_forward(x, self.inc3b)
        x = F.max_pool2d(x, 3, 2, ceil_mode=True)

        x = self._inception_forward(x, self.inc4a)
        x = self._inception_forward(x, self.inc4b)
        x = self._inception_forward(x, self.inc4c)
        x = self._inception_forward(x, self.inc4d)
        x = self._inception_forward(x, self.inc4e)
        x = F.max_pool2d(x, 3, 2, ceil_mode=True)

        x = self._inception_forward(x, self.inc5a)
        x = self._inception_forward(x, self.inc5b)
        x = self.pool(x)
        x = x.view((x.size(0), -1))
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _inception_forward(self, x, inception):
        conv1_1 = inception['1'](x)
        conv3_3 = inception['3'](x)
        conv5_5 = inception['5'](x)
        conv_p = inception['p'](x)
        output = torch.cat([conv1_1, conv3_3, conv5_5, conv_p], dim=1)
        return output

    def _make_inception(self, in_channels, out_channels_1_1, reduce_channels_3_3, out_channels_3_3,
                        reduce_channels_5_5, out_channels_5_5, pool_proj, check):
        assert out_channels_1_1 + out_channels_3_3 + out_channels_5_5 + pool_proj == check
        inception = {}
        inception['1'] = nn.Sequential(*[nn.Conv2d(in_channels, out_channels_1_1, kernel_size=1),
                                         nn.BatchNorm2d(out_channels_1_1),
                                         nn.ReLU(inplace=True)])
        inception['3'] = nn.Sequential(*[nn.Conv2d(in_channels, reduce_channels_3_3, kernel_size=1),
                                         nn.BatchNorm2d(reduce_channels_3_3),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(reduce_channels_3_3, out_channels_3_3, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(out_channels_3_3),
                                         nn.ReLU(inplace=True)])
        inception['5'] = nn.Sequential(*[nn.Conv2d(in_channels, reduce_channels_5_5, kernel_size=1),
                                         nn.BatchNorm2d(reduce_channels_5_5),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(reduce_channels_5_5, out_channels_5_5, kernel_size=5, padding=2),
                                         nn.BatchNorm2d(out_channels_5_5),
                                         nn.ReLU(inplace=True)])
        inception['p'] = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
                                         nn.Conv2d(in_channels, pool_proj, kernel_size=1),
                                         nn.BatchNorm2d(pool_proj),
                                         nn.ReLU(inplace=True)])
        inception = nn.ModuleDict({key: value for key, value in inception.items()})
        return inception

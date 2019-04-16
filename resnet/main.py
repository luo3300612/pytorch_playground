import torch
import torch.nn as nn


class ResNet34(nn.Module):
    def __init__(self, H, W, in_channel=3, num_classes=10):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.block1 = ResNetBlock(64, 64)
        self.block2 = ResNetBlock(64, 64)
        self.block3 = ResNetBlock(64, 128)
        self.block4 = ResNetBlock(128, 128, down_sample=True)
        self.block5 = ResNetBlock(128, 128)
        self.block6 = ResNetBlock(128, 128)
        self.block7 = ResNetBlock(128, 256)
        self.block8 = ResNetBlock(256, 256, down_sample=True)
        self.block9 = ResNetBlock(256, 256)
        self.block10 = ResNetBlock(256, 256)
        self.block11 = ResNetBlock(256, 256)
        self.block12 = ResNetBlock(256, 256)
        self.block13 = ResNetBlock(256, 512)
        self.block14 = ResNetBlock(512, 512, down_sample=True)
        self.block15 = ResNetBlock(512, 512)
        self.block16 = ResNetBlock(512, 512)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.fc = nn.Linear(int(H * W / (2 ** 10) * 512), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample=False):
        super(ResNetBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1 + int(down_sample), padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        shape = 1, self.out_channel - self.in_channel, x.shape[2], x.shape[3]
        print("x shape:", x.shape)
        print("shape:", shape)
        out = out + torch.cat((x, torch.zeros(shape)), dim=1)
        out = self.relu(out)
        return out


net = ResNet34(224, 224, in_channel=3, num_classes=10)

x = torch.randn((1, 3, 224, 224))
out = net(x)
print(out.shape)

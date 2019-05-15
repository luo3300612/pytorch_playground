import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter
import sys
import torch.nn.functional as F
from utils import OutPutUtil

# class ResNet18(nn.Module):
#     def __init__(self, H, W, in_channel=3, num_classes=10):
#         super(ResNet18, self).__init__()
#         self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(2, 2)
#         self.block1 = ResNetBlock(64, 64, 2, down_sample=False)
#         self.block2 = ResNetBlock(64, 128, 2)
#         self.block3 = ResNetBlock(128, 256, 2)
#         self.block4 = ResNetBlock(256, 512, 2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = Flatten()
#         self.fc = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.avgpool(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x


# class ResNet34(nn.Module):
#     def __init__(self, H, W, in_channel=3, num_classes=10):
#         super(ResNet34, self).__init__()
#         self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3)
#         self.maxpool = nn.MaxPool2d(2, 2)
#         self.block1 = ResNetBlock(64, 64, 3, down_sample=False)
#         self.block2 = ResNetBlock(64, 128, 4)
#         self.block3 = ResNetBlock(128, 256, 6)
#         self.block4 = ResNetBlock(256, 512, 3)
#         self.avgpool = nn.AvgPool2d((H // (2 ** 5), W // (2 ** 5)))
#         self.flatten = Flatten()
#         self.fc = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool(x)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.avgpool(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x


class ResNet20(nn.Module):
    """
    后面有bn bias可以是Fasle
    nn.AdaptiveAvgPool2d好用
    nn.Sequential好用
    """

    def __init__(self, in_channel, num_classes=10):
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

    def _make_layer(self, in_channel, out_channel, num_block, downsample=False):
        layers = []
        if downsample:
            downsample = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2)
        else:
            downsample = None

        layers.append(ResNetBlock_new(in_channel, out_channel, downsample))
        for i in range(1, num_block):
            layers.append(ResNetBlock_new(out_channel, out_channel))

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


class ResNet18_new(nn.Module):
    """
    后面有bn bias可以是Fasle
    nn.AdaptiveAvgPool2d好用
    nn.Sequential好用
    """

    def __init__(self, in_channel, num_classes=1000):
        super(ResNet18_new, self).__init__()
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
            downsample = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2)
        else:
            downsample = None

        layers.append(ResNetBlock_new(in_channel, out_channel, downsample))
        for i in range(1, num_block):
            layers.append(ResNetBlock_new(out_channel, out_channel))

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


class ResNetBlock_new(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=None):
        super(ResNetBlock_new, self).__init__()
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
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out = out + x  # TODO together batch normalization??
        out = self.bn2(out)
        out = self.relu(out)
        return out


# class ResNetBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, block_num, down_sample=True):
#         super(ResNetBlock, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.block_num = block_num
#         self.down_sample = down_sample
#
#         self.convx = None
#         if down_sample:
#             self.convx = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False)
#
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(2, 2)
#         self.layers = []
#         self.layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1 + int(down_sample),
#                                      padding=1))
#         for i in range(1, self.block_num):
#             if i == 0:
#                 self.__setattr__(f"conv{i + 1}_1",
#                                  nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1 + int(down_sample),
#                                            padding=1))
#             else:
#                 self.__setattr__(f"conv{i + 1}_1",
#                                  nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1))
#             if i != block_num - 1:
#                 self.__setattr__(f"conv{i + 1}_2", nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
#             else:
#                 self.__setattr__(f"conv{i + 1}_2", nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
#
#     def forward(self, x):
#
#         for i in range(self.block_num):
#             out = self.relu(self.__getattr__(f"conv{i + 1}_1")(x))
#             out = self.__getattr__(f"conv{i + 1}_2")(out)
#             if i is 0 and self.down_sample:
#                 x = self.maxpool(x)
#                 x = F.pad(x, (0, 0, 0, 0, 0, self.out_channel - self.in_channel))
#                 out = out + x
#                 # shape = x.shape[0], self.out_channel - self.in_channel, x.shape[2], x.shape[3]
#                 # out = out + torch.cat((x, torch.zeros(shape).to(device)),
#                 #                       dim=1)  # fixme: when load model in notebook, device is undefined
#             else:
#                 out = out + x
#             out = self.relu(out)
#             x = out
#         return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def adjust_learning_rate(optimizer, iteration, init_lr=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if iteration <= 32000:
        lr = init_lr
    elif 32000 < iteration <= 48000:
        lr = init_lr / 10
    else:
        lr = init_lr / 100
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# net = ResNet18(224, 224, in_channel=3, num_classes=10)
#
# x = torch.randn((20, 3, 224, 224))
# out = net(x)
# print(out.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch ResNet CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training and testing (default: 128)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-path', type=str, default='./train.log',
                        help='path to save log file (default: ./train.log)')
    args = parser.parse_args()

    monitor = OutPutUtil(True, True, args.log_path)

    batch_size = args.batch_size

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_data = torchvision.datasets.CIFAR10(root='./data/',
                                              train=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                              ]),
                                              download=True)
    test_data = torchvision.datasets.CIFAR10(root='./data/',
                                             train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ]),
                                             download=True)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=1)

    # net = ResNet18(32, 32, 3, 10).to(device)
    net = ResNet20(3, 10).to(device)
    # net = torchvision.models.resnet18(False, **{"num_classes": 10}).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    writer = SummaryWriter()

    iter_idx = 0
    n_iter = 64000
    val_interval = 100
    print_interval = 10

    best_test_loss = 3
    while True:
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            iter_idx += 1
            lr = adjust_learning_rate(optimizer, iter_idx, init_lr=0.01)
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_idx % print_interval == 0:
                monitor.speak('Iter: {}/{}\tLoss:{:.6f}\tLR: {}'.format(iter_idx, n_iter, loss.item(), lr))
            writer.add_scalar("train/train_loss", loss.item(), iter_idx)

            if iter_idx % val_interval == 0:
                test_loss = 0.0
                with torch.no_grad():
                    acc = 0.0
                    for data, target in test_loader:
                        data = data.to(device)
                        target = target.to(device)
                        output = net(data)
                        pred_label = torch.argmax(output, dim=1)
                        acc += torch.sum(pred_label == target).item()
                        loss = criterion(output, target)
                        test_loss += loss.item() * data.shape[0]
                    acc = acc / len(test_data)
                    test_loss = test_loss / len(test_data)

                    monitor.speak('Test Loss: {:.6f},acc:{:.4f}'.format(test_loss, acc))
                    writer.add_scalar("train/test_loss", test_loss, iter_idx)
                    writer.add_scalar("train/acc", acc, iter_idx)
                if test_loss < best_test_loss:
                    torch.save(net.state_dict(), r"./result/model{}".format(iter_idx))
                    monitor.speak("test loss: {:.6f} < best: {:.6f},save model".format(test_loss, best_test_loss))
                    best_test_loss = test_loss

            if iter_idx == n_iter:
                monitor.speak("Done")
                break

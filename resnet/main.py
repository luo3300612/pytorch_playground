import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter
import sys

device = torch.device("cpu")


class ResNet18(nn.Module):
    def __init__(self, H, W, in_channel=3, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.block1 = ResNetBlock(64, 64, 2, down_sample=False)
        self.block2 = ResNetBlock(64, 128, 2)
        self.block3 = ResNetBlock(128, 256, 2)
        self.block4 = ResNetBlock(256, 512, 2)
        self.avgpool = nn.AvgPool2d((H // (2 ** 5), W // (2 ** 5)))
        self.flatten = Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, H, W, in_channel=3, num_classes=10):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.block1 = ResNetBlock(64, 64, 3, down_sample=False)
        self.block2 = ResNetBlock(64, 128, 4)
        self.block3 = ResNetBlock(128, 256, 6)
        self.block4 = ResNetBlock(256, 512, 3)
        self.avgpool = nn.AvgPool2d((H // (2 ** 5), W // (2 ** 5)))
        self.flatten = Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, block_num, down_sample=True):
        super(ResNetBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block_num = block_num
        self.down_sample = down_sample

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        for i in range(self.block_num):
            if i == 0:
                self.__setattr__(f"conv{i + 1}_1",
                                 nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1 + int(down_sample),
                                           padding=1))
            else:
                self.__setattr__(f"conv{i + 1}_1",
                                 nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1))
            if i != block_num - 1:
                self.__setattr__(f"conv{i + 1}_2", nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
            else:
                self.__setattr__(f"conv{i + 1}_2", nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))

    def forward(self, x):

        for i in range(self.block_num):
            out = self.relu(self.__getattr__(f"conv{i + 1}_1")(x))
            out = self.__getattr__(f"conv{i + 1}_2")(out)
            if i is 0 and self.down_sample:
                x = self.maxpool(x)
                shape = x.shape[0], self.out_channel - self.in_channel, x.shape[2], x.shape[3]
                out = out + torch.cat((x, torch.zeros(shape).to(device)),
                                      dim=1)  # fixme: when eval, device is undefined
            else:
                out = out + x
            out = self.relu(out)
            x = out
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if iteration <= 32000:
        lr = 0.1
    elif 32000 < iteration <= 48000:
        lr = 0.01
    else:
        lr = 0.001
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
                        help='input batch size for training and testing (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()

    batch_size = args.batch_size

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_data = torchvision.datasets.CIFAR10(root='./data/',
                                              train=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                              ]),
                                              download=False)
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

    net = ResNet18(32, 32, 3, 10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.5)

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
            lr = adjust_learning_rate(optimizer, iter_idx)
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_idx % print_interval == 0:
                print(f'Iter: {iter_idx}/{n_iter}\tLoss:{loss.item():.6f}\tLR: {lr}')
            writer.add_scalar("train/train_loss", loss.item(), iter_idx)

            if iter_idx % val_interval == 0:
                test_loss = 0.0
                with torch.no_grad():
                    for data, target in test_loader:
                        data = data.to(device)
                        target = target.to(device)
                        output = net(data)
                        loss = criterion(output, target)
                        test_loss += loss.item() * data.shape[0]
                    print(f'Iter: {iter_idx}/{n_iter}\tTest Loss: {test_loss / len(test_data):.6f}')
                    writer.add_scalar("train/test_loss", test_loss / len(test_loader), iter_idx)
                if test_loss / len(test_data) < best_test_loss:
                    torch.save(net, r"E:\pycharmprojects\pytorch_playground\resnet\result\model{}".format(iter_idx))
                    print(f"test loss: {test_loss / len(test_data)} < best: {best_test_loss},save model")
                    best_test_loss = test_loss / len(test_data)

            if iter_idx == n_iter:
                print("Done")
                break

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.optim as optim


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
                out = out + torch.cat((x, torch.zeros(shape)), dim=1)
            else:
                out = out + x
            out = self.relu(out)
            x = out
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


# net = ResNet18(224, 224, in_channel=3, num_classes=10)
#
# x = torch.randn((20, 3, 224, 224))
# out = net(x)
# print(out.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch ResNet CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training and testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train(default:100)')
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
                                                  transforms.Normalize((0.5,), (0.5,))
                                              ]),
                                              download=True)
    test_data = torchvision.datasets.MNIST(root='./data/',
                                           train=False,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5,), (0.5,))
                                           ]),
                                           download=True)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)

    net = ResNet18(32, 32, 3, 10).to(device)
    n_epoch = args.epochs

    criterion = nn.LogSoftmax()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(n_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch:{epoch}/{n_epoch}\tBatch:{batch_idx}\tLoss:{loss.item():.6f}')

        with torch.no_grad():
            test_loss = 0.0
            for data, target in test_loader:
                output = net(data)
                loss = criterion(output, target)
                test_loss += loss.item() * batch_size
            print(f'Epoch:{epoch} Finished\t Test Loss{test_loss / len(test_loader):.6f}')

from utils.utils import OutPutUtil
import torch
import torchvision.datasets.cifar
import argparse
import torchvision.transforms as transforms
import multiprocessing
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import itertools
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter


class LeNetPool(nn.Module):
    def __init__(self, input_channel):
        super(LeNetPool, self).__init__()
        self.pool = nn.AvgPool2d(2, 2)
        self.w = nn.Parameter(torch.empty(input_channel, 1, 1))
        self.b = nn.Parameter(torch.empty(input_channel, 1, 1))

        nn.init.constant_(self.w, 4)
        nn.init.constant_(self.b, 0)

    def forward(self, x):
        x = self.pool(x)
        x = x * self.w + self.b
        return x


class LeConv2d(nn.Module):
    def __init__(self):
        super(LeConv2d, self).__init__()
        self.convs1 = [nn.Conv2d(3, 1, 5) for _ in range(6)]
        self.convs2 = [nn.Conv2d(4, 1, 5) for _ in range(9)]
        self.last_conv = nn.Conv2d(6, 1, 5)
        self.combinations = []
        for i in range(6):
            self.combinations.append([i % 6, (i + 1) % 6, (i + 2) % 6])
        for i in range(6):
            self.combinations.append(([i % 6, (i + 1) % 6, (i + 2) % 6, (i + 3) % 6]))
        self.combinations.append([0, 1, 3, 4])
        self.combinations.append([1, 2, 4, 5])
        self.combinations.append([0, 2, 3, 5])

    def forward(self, x):
        output = torch.empty((x.size(0), 16, 10, 10))
        for i in range(15):
            inp = x[:, self.combinations[i]]
            if i < 6:
                output[:, i] = self.convs1[i](inp)
            else:
                output[:, i] = self.convs2[i](inp)
        output[:, 15] = self.last_conv(x)
        return output


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = LeNetPool(input_channel=6)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = LeConv2d()
        self.pool2 = LeNetPool(input_channel=16)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.A = torch.Tensor(1.7159)
        self.S = torch.Tensor(1)
        self.tanh = nn.Tanh()
        # self.w = nn.

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.sigmoid(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.sigmoid(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.A * self.tanh(self.S * x)
        return x


class NewLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(NewLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.maxpool(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = x.view((x.size(0), -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def save_checkpoint(epoch, model, optimizer, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict,
        'loss': loss
    }, save_path)


def adjust_learning_rate(optimizer, iteration, n_iter, init_lr=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if iteration / n_iter <= 0.5:
        lr = init_lr
    elif 0.5 < iteration / n_iter <= 0.75:
        lr = init_lr / 10
    else:
        lr = init_lr / 100
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch LeNet CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training and testing (default: 128)')
    parser.add_argument('--val-interval', type=int, default=1000,
                        help='val interval (default: 1000)')
    parser.add_argument('--n-iter', type=int, default=64000,
                        help='total iteration (default: 64000)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--m', type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--adam', action='store_true', default=False,
                        help='use adam')
    parser.add_argument('--log-path', type=str, default='./train.log',
                        help='path to save log file (default: ./train.log)')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save checkpoint (default:False)')
    parser.add_argument('--save_path', type=str, default='./result',
                        help='model save path (default: ./result')
    args = parser.parse_args()

    monitor = OutPutUtil(True, True, args.log_path)
    monitor.speak(args)
    writer = SummaryWriter()

    use_cuda = not args.no_cude and torch.cuda.is_avaliable()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_data = torchvision.datasets.CIFAR10(root='../resnet/data/',
                                              train=True,
                                              transform=transforms.Compose([
                                                  transforms.RandomCrop(32, 4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010)),
                                              ]),
                                              download=True)
    test_data = torchvision.datasets.CIFAR10(root='../resnet/data/',
                                             train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                      (0.2023, 0.1994, 0.2010))
                                             ]),
                                             download=True)

    batch_size = args.batch_size
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)

    net = NewLeNet(10).to(device)
    criterion = nn.CrossEntropyLoss()
    init_lr = args.lr
    lr = args.lr
    if args.adam:
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args.m)

    n_iter = args.n_iter
    val_interval = args.val_interval
    print_interval = 10

    best_test_loss = 4
    iter_idx = 0
    net.train()
    while True:
        for batch_idx, (data, target) in enumerate(train_loader):
            lr = adjust_learning_rate(optimizer, iter_idx, n_iter, init_lr=init_lr)
            iter_idx += 1
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_idx % print_interval == 0:
                monitor.speak('Iter: {}/{}\tLoss:{:.6f}\tLR: {}'.format(iter_idx, n_iter, loss.item(), lr))
            writer.add_scalar("train/train_loss", loss.item(), iter_idx)

            if iter_idx % val_interval == 0:
                net.eval()
                test_loss = 0.0
                acc = 0.0
                with torch.no_grad():
                    for data, target in enumerate(test_loader):
                        data = data.to(device)
                        target = target.to(device)
                        output = net(data)
                        pred_label = torch.argmax(output, 1)
                        acc += torch.sum(pred_label == target).item()
                        loss = criterion(output, target)
                        test_loss += loss.item() * data.shape[0]
                    acc = acc / len(test_data)
                    test_loss = test_loss / len(test_data)
                monitor.speak(())
                monitor.speak('Test Loss: {:.6f},acc:{:.4f}'.format(test_loss, acc))
                writer.add_scalar("train/test_loss", test_loss, iter_idx)
                writer.add_scalar("train/acc", acc, iter_idx)
            if test_loss < best_test_loss:
                if args.save:
                    save_checkpoint(iter_idx, net, optimizer, loss.item(), args.save_path)
                monitor.speak("test loss: {:.6f} < best: {:.6f},save if asked".format(test_loss, best_test_loss))
                best_test_loss = test_loss
            net.train()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter
import sys
sys.path.append('../')
import torch.nn.functional as F
from utils.utils import OutPutUtil
import numpy as np
from torch.nn import DataParallel
import time
import models
from torchvision.models import vgg19_bn
from pathlib import Path


# 为了根据iteration的次数进行输出和test，就不用函数的形式包裹train和test了

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


def save_checkpoint(epoch, model, optimizer, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict,
        'loss': loss
    }, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch CIFAR10 Example')
    parser.add_argument('--model', type=str, default='vgg',
                        help='classifier model (default: vgg)')
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
    parser.add_argument('--save', action='store_true', default=False,
                        help='save checkpoint (default:False)')
    parser.add_argument('--checkpoint_path', type=str, default='./result',
                        help='model save path (default: ./result)')
    parser.add_argument('--val', action='store_true', default=False,
                        help='val mode (default: False)')
    parser.add_argument('--no-log', action='store_true', default=False,
                        help='no output log file (default: False)')
    args = parser.parse_args()

    if not args.q:
        monitor = OutPutUtil(True, True, log_file=str(Path(args.checkpoint_path, './train.log')))
    else:
        monitor = OutPutUtil(True, log=False, log_file=str(Path(args.checkpoint_path, './train.log')))
    monitor.speak(args)
    writer = SummaryWriter()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    sys.path.append('./')
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

    net = models.setup(args)
    print(net)
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
        torch.cuda.synchronize()
        start = time.time()
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
                torch.cuda.synchronize()
                end = time.time()
                monitor.speak('Iter: {}/{}\tLoss:{:.6f}\tLR: {}\ttime/batch'.format(iter_idx, n_iter, loss.item(), lr,
                                                                                    (end - start) / batch_size))
            writer.add_scalar("train/train_loss", loss.item(), iter_idx)

            if iter_idx % val_interval == 0:
                net.eval()
                test_loss = 0.0
                acc = 0.0
                with torch.no_grad():
                    for data, target in test_loader:
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
                        save_checkpoint(iter_idx, net, optimizer, loss.item(), args.checkpoint_path)
                        monitor.speak("test loss: {:.6f} < best: {:.6f},save".format(test_loss, best_test_loss))
                    else:
                        monitor.speak("test loss: {:.6f} < best: {:.6f},not save".format(test_loss, best_test_loss))
                    best_test_loss = test_loss
                net.train()
        if iter_idx > n_iter:
            monitor.speak("Done")
            break

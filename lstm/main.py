import argparse
import torch
from dataloader import get_loader
from model import myLSTM
import torch.nn as nn
import torch.optim as optim
import time


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.9 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-h5', type=str,
                        default='/home/luo3300612/Workspace/PycharmWS/pytorch_playground/lstm/data/data.h5')
    parser.add_argument('--data-json', type=str,
                        default='/home/luo3300612/Workspace/PycharmWS/pytorch_playground/lstm/data/data.json')
    parser.add_argument('--batch-size', type=int,
                        default=32)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--vocab-size', type=int, default=2200)
    parser.add_argument('--max-length', type=int, default=20)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--m', type=float, default=0.9)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--print_every', type=int, default=50)
    opt = parser.parse_args()

    use_cuda = opt.cuda and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    train_loader, test_loader = get_loader(opt)
    net = myLSTM(opt).to(device)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.m)

    iteration = 0
    acc_print = 0
    loss_print = 0
    interval = 0
    best_acc = 0
    net.train()
    for epoch in range(opt.max_epoch):
        # adjust_learning_rate(optimizer, epoch,opt)
        start = time.time()
        for i, (tokens, label) in enumerate(train_loader):
            tokens = tokens.to(device).long()
            label = label.to(device).long()
            output = net(tokens)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = torch.argmax(output, dim=1)
            acc_print += torch.sum(output == label).item()
            loss_print += loss.item() * label.shape[0]
            interval += label.shape[0]

            if iteration % opt.print_every is 0:
                print('epoch {}, iter {}, loss {:.4f}, ACC {:.2f}, time {:.4f},'.format(epoch, iteration,
                                                                                        loss_print / interval,
                                                                                        acc_print * 100 / interval,
                                                                                        time.time() - start))
                acc_print = 0
                loss_print = 0
                interval = 0
                start = time.time()
            iteration += 1

        print('start eval...')
        net.eval()
        acc_test = 0
        loss_test = 0
        test_interval = 0
        test_iteration = 0
        with torch.no_grad():
            for i, (tokens, label) in enumerate(test_loader):
                tokens = tokens.to(device).long()
                label = label.to(device).long()
                output = net(tokens)
                loss = criterion(output, label)

                output = torch.argmax(output, dim=1)
                acc_test += torch.sum(output == label).item()
                loss_test += loss.item() * label.shape[0]
                test_interval += label.shape[0]
                test_iteration += 1
                if test_iteration % opt.print_every is 0:
                    print('Testing... {}/{} {:.2f}%,'.format(test_interval, len(test_loader.dataset),
                                                             test_interval * 100 / len(test_loader.dataset)))
            if acc_test / test_interval * 100 > best_acc:
                best_acc = acc_test / test_interval * 100
            print('<Test> Epoch {}, Loss {:.4f}, ACC {:.2f}, BEST ACC {:.2f}'.format(epoch,
                                                                                     loss_test / test_interval,
                                                                                     acc_test / test_interval * 100,
                                                                                     best_acc))

        net.train()

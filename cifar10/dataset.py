import torchvision
import torchvision.transforms as transforms
import multiprocessing
from torch.utils.data import DataLoader


def get_loader(args):
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

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=multiprocessing.cpu_count(),
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=multiprocessing.cpu_count(),
                             pin_memory=True)
    return train_loader, test_loader, train_data, test_data

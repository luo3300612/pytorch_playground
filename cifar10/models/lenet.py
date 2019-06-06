import sys

sys.path.append('../')

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
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = x.view((x.size(0), -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

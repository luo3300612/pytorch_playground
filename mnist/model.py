import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_planes, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, 8, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.flatten = Flatten()
        self.fc = nn.Linear(7 * 7 * 16, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.flatten = Flatten()
        self.fc1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

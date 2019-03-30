import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt

train_data = torchvision.datasets.MNIST(root='./data/',
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=False)
test_data = torchvision.datasets.MNIST(root='./data/',
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=False)


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = FullyConnectedNet(10, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

print(train_data.data.size())
print(test_data.data.size())
plt.imshow(train_data.data[0], cmap='gray')
plt.show()

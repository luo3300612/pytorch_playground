import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from model import Net, CNN
from tensorboardX import SummaryWriter
import numpy as np

batch_size = 10
learning_rate = 0.01
momentum = 0.9

writer = SummaryWriter()

train_data = torchvision.datasets.MNIST(root='./data/',
                                        train=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]),
                                        download=False)
test_data = torchvision.datasets.MNIST(root='./data/',
                                       train=False,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]),
                                       download=False)

train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)
test_loader = DataLoader(test_data,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)

net = CNN(1, 10)
net.train()

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

n_epoch = 100
num_of_batch = 100

for epoch in range(n_epoch):
    running_loss = 0.0
    for batch_i, data in enumerate(train_loader):
        if batch_i == num_of_batch:
            break

        imgs = data[0]
        labels = data[1]

        outputs = net(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    test_loss = 0.1
    acc = 0.0
    for batch_i, data in enumerate(test_loader):
        if batch_i == num_of_batch / 5:
            break
        imgs = data[0]
        labels = data[1]
        with torch.no_grad():
            outputs = net(imgs)
            output_labels = torch.argmax(outputs, dim=1)

            acc += torch.sum(output_labels == labels).item()
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    # writer.add_scalar('scalar/const', 1, epoch)
    writer.add_scalars('cnn_scalar/loss', {'running_loss': running_loss / num_of_batch,
                                       'test_loss': test_loss / num_of_batch * 5}, epoch)
    writer.add_scalar('cnn_scalar/acc', acc / num_of_batch * 5 * 100/batch_size, epoch)
    print(
        f"""epoch:{epoch + 1},Avg.loss:{running_loss / num_of_batch :.5f},test loss:{test_loss / num_of_batch * 5 :.5f},acc:{acc / num_of_batch * 5 * 100 / batch_size:.2f}%""")

writer.export_scalars_to_json("./all_scalars.json")
writer.close()

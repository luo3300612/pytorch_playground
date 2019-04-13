import torch
import torch.optim as optim
import numpy as np
from functools import reduce
from operator import add
import matplotlib.pyplot as plt
import torch.nn as nn
from tensorboardX import SummaryWriter

writer = SummaryWriter()

coefficients = [0, 0, -3, 1, 1]
poly = lambda x: reduce(add, [c * x ** i for i, c in enumerate(coefficients)])

X = np.linspace(-2, 2, 100)
Y = poly(X)

ymean = np.mean(Y)
ystd = np.std(Y)
Y = (Y - ymean) / ystd

xmean = np.mean(X)
xstd = np.std(X)
X = (X - xmean) / xstd


# plt.plot(X, y)
# plt.show()

class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x


if __name__ == '__main__':

    net = FullyConnectedNet()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    n_epoch = 1000

    for epoch in range(n_epoch):
        epoch_loss = 0.0
        for i in range(X.shape[0]):
            x = torch.tensor([X[i] ** j for j in range(1, 5)])
            y = torch.tensor(Y[i])

            output = net(x)
            loss = criterion(output, y)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch:{epoch + 1},loss:{epoch_loss}")
        writer.add_scalar('data/epoch_loss', epoch_loss, epoch + 1)

    torch.save(net, "model")

    writer.close()

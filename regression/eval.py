import torch
from main import FullyConnectedNet, poly
import numpy as np
import matplotlib.pyplot as plt

net = torch.load("./model")

xs = np.linspace(-2, 2, 100)

ground_truth = poly(xs)
pred = np.zeros((xs.shape[0],))

for i in range(xs.shape[0]):
    pred[i] = net(torch.tensor([xs[i]])).item()

plt.plot(xs, ground_truth, color='r')
plt.plot(xs, pred, color='b')
plt.show()
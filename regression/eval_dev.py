import torch
from dev import FullyConnectedNet, poly,ymean,ystd
import numpy as np
import matplotlib.pyplot as plt


net = torch.load("./model")

xs = np.linspace(-2, 2, 100)
x_norm = (xs-np.mean(xs))/np.std(xs)
print(x_norm)
ground_truth = poly(xs)
pred = np.zeros((xs.shape[0],))

for i in range(xs.shape[0]):
    pred[i] = net(torch.tensor([x_norm[i]**j for j in range(1,5)])).item()*ystd + ymean

plt.plot(xs, ground_truth, color='r')
plt.plot(xs, pred, color='b')
plt.show()
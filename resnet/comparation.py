from torchvision.models.resnet import resnet18
import torch

config = {"num_classes": 10}

net = resnet18(**config)

x = torch.randn(size=[5, 3, 224, 64])

output = net(x)

print(output.shape)
from torchvision.models.resnet import resnet18
import torch
from main import ResNet20

config = {"num_classes": 10}

# net = resnet18(**config)

net = ResNet20(3,10)

x = torch.randn(size=[5, 3, 32, 32])

output = net(x)

print(output.shape)
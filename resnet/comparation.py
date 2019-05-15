from torchvision.models.resnet import resnet18
import torch
from main import ResNet18_new

config = {"num_classes": 10}

# net = resnet18(**config)

net = ResNet18_new(3,10)

x = torch.randn(size=[5, 3, 224, 64])

output = net(x)

print(output.shape)
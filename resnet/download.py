import torchvision
import torchvision.transforms as transforms

train_data = torchvision.datasets.CIFAR10(root='./data/',
                                          train=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]),
                                          download=True)
test_data = torchvision.datasets.MNIST(root='./data/',
                                       train=False,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]),
                                       download=True)

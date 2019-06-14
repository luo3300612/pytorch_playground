from .lenet import *
from .vgg import *
from .alexnet import *
from .resnet import *


def setup(opt):
    if opt.model == 'vgg':
        model = VGG()
    elif opt.model == 'lenet':
        model = NewLeNet()
    elif opt.model == 'resnet':
        model = ResNet20()
    elif opt.model == 'alexnet':
        model = AlexNet()
    else:
        raise NotImplementedError
    return model

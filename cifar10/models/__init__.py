from .lenet import *
from .vgg import *


def setup(opt):
    if opt.model == 'vgg':
        model = VGG()
    elif opt.model == 'lenet':
        model = NewLeNet()
    else:
        raise NotImplementedError
    return model

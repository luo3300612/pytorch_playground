from .lenet import *
from .vgg import *
from .alexnet import *
from .resnet import *
from .googlenet import *
import torchvision


def setup(opt):
    if opt.model == 'vgg':
        model = VGG()
    elif opt.model == 'lenet':
        model = NewLeNet()
    elif opt.model == 'resnet':
        model = ResNet20()
    elif opt.model == 'alexnet':
        model = AlexNet()
    elif opt.model == 'googlenet':
        model = GoogLeNet()
    elif opt.model == 'torch_vgg':
        kwargs = {'num_classes': 10}
        model = torchvision.models.vgg16(**kwargs)
    else:
        raise NotImplementedError
    return model

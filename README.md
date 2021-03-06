# Pytorch Playground
## MNIST
### 准备
数据下载[地址](http://yann.lecun.com/exdb/mnist/)
### 问题
#### SGD的参数lr和momentum怎么设置，优化的公式是什么
#### 使用jupyter-notebook还是什么来构建项目？
选择1：使用之前FKP项目的结构，在.py文件中写网络结构和数据类，然后利用jupyter-notebook的实时导入来写
* (+)模块区分明确，可以记笔记，方便查看数据集
* (-)比较学术和笔记，不方便当脚本使用，打开麻烦
选择2：只使用Jupyter-notebook来写
* 优缺点和1一样，更改方便，但模块化低
选择3：全写在.py文件中
* (+)方便使用
* (-)难记笔记
目前方案：
* 先在jupyter-notebook里写完整的版本
* 然后在.py文件中重新构建项目
#### 对MNIST做或不做除以255有什么影响
原数据就是0~1之间的浮点数，不需要再除以255
#### 使用或不使用momentum有什么影响
小学习率和momentum是好的组合
#### SGD with momentum的公式是啥
原来的公式是
```python
x += learning_rate * gradient
```
加上momentum公式是
```python
v = momentum * v - learning_rate * gradient
x += v
```
#### net在什么时候初始化参数？net.train还是什么？
网络在init方法后会自动初始化参数，net.train与net.eval是对特殊层（如bn、dropout）在训练和测试时切换表现的选项
#### tensor.data和tensor.item有啥区别
`tensor.data`虽然可以使用，但是文档中查不到，且返回还是一个tensor，而tensor.item()则返回python类型的数值，要求tensor是一个一维张量
#### 输入图片的tensor是uint8还是float有啥区别
不知道
#### 各种网络不同层的结构分别有什么区别，它们的性能怎样
### 坑
dataset类的transform参数后要打括号，比如transform = ToTensor()，否则会报object...错误

### 笔记
[torch.no_grad](https://pytorch.org/docs/stable/autograd.html?highlight=no_grad#torch.autograd.no_grad)上下文管理器，节约计算资源
CONV-relu-pool
### TODO
训练过程记录入log中 
可视化训练过程 done
保存模型 done

## Regression
### 问题
* 自己打包的batch好像无法被forward接受为输入？
    可以，第一维为batch_size即可
* 使用怎样的损失函数才比较合理？
* Adam的不稳定的情况,在loss降低到极限的时候会出现不稳定的情况
* 
### 坑
模型的拟合效果并不好，可以调整的有以下几个方面
* 数据集
* 模型复杂度
* 损失函数
* 优化方法
* 超参数
* 归一化

通过归一化之后，loss以及可以降低到还不错的位置，归一化可以解决部分难优化的问题
### 发现
* x的正则化可以使更大的学习率能够被使用
* 为了达到类似的训练效果Adam的学习率常常比SGD小（十分之一左右）
* 更深的模型lr需要更小
* SMOOTHL1Loss比MSE计算慢的多,shit，是因为用SMOOTHL1loss的时候每插电源....
* Adam虽然快，但是到最后不稳定

## ResNet
使用ResNet拟合CIFAR10
### 训练参数
这是来自[ResNet论文](https://arxiv.org/abs/1512.03385)的CIFAR10训练参数
* mini batch 128
* 初始学习率0.1，在32k和48k次迭代时除以10
* 在64k次迭代时终止
### 问题
* resnet在进行skip connection的时候会出现不仅仅是channel增加（用0padding或投影
解决），而且会出现feature map size减小的情况，对此，论文上给出的是when the 
short cuts go across feature maps of two sizes, they are performed with 
a stride of 2，即便使用了stride of 2，那是要用maxpool吗？还是要随机选一个元素
呢？
先用maxpool试试

正确做法是使用1\*1的卷积同时进行channel和feature map的变换
* skip connection的时候，x是和out一起做bn还是？
x和bn分别做bn然后加到一起做relu，一开始我用一起做结果训练的acc只有81%
* cifar10 的augmentation:4padding + random crop((32,32)) + random horizontal flip
* lr用错了，weight decay没加、初始化方法没用、normalize不对，acc约0.85左右
* 加上了weight decay和正确的lr，acc能到0.89
* 加上推荐的norm之后竟然只有0.87？？？
* 加上推荐的norm和init方法后大概是0.88左右
* net.train()，0.9199，解决问题

### 源码对比结果
* 源码中在relu前使用了batch normalization层
* 源码中第一个下采样的maxpool是3\*3 padding=1 stride=2的，而我的是2\*2 padding=0 stride=2的，这有啥影响
* 源码中的relu使用的是就地操作nn.ReLU(inplace=True)，这样可以节省一点内存
* 源码中卷积使用了bias=False参数
* 源码中的avgpool使用的是Adaptiveavgpool，只要给输出的size就可以
* 源码中奇怪的初始化方法
* 我conv1后面竟然没有做relu
* 源码中为了让x通过skip connection，通过1*1的卷积使得x的channel对应，并通过设置stride=2使得空间维度对应，这实际上是论文中对应的b类方案
* 源码中使用列表，然后用nn.Sequential(*layers)来动态加入层


### 经验
* 后面有bn bias可以是Fasle
* nn.AdaptiveAvgPool2d好用
* nn.Sequential好用


## LeNet
### 结构
* 输入：32×32×3
* 卷积：5×5 -> 28×28×6
* 下采样：2×2 每个区域中加起来，然后线性变换一下 -> 14×14×6
* sigmoid -> 14×14×6
* 卷积： 5×5 -> 10×10×16 ,  very starnge，原因，1：减少连接数，2：打破对称性3
* 下采样：与前一个一样 -> 5×5×16
* sigmoid(?) -> 5×5×16
* 卷积：5×5 ->1×1×120
* 全连接： -> 1×84
* 非线性:  Atanh(Sx) -> 1×84
* 径向基：
## Pytorch Tutorials
* pytorch是动态图，tensorflow是静态图，静态图一次生成多次使用，动态图在使用过程中动态生成，静态图利于优化，动态图利于灵活的流程控制
* tf中kears,TensorFlow-Flim和TFLearn提供了高层抽象，pytorch中，这些高层抽象在torch.nn中

## AlexNet
### 特点
5CONV+3FC，其中前两个FC有dropout
### difference
具体实现采用了现代网络设方法，与原版不同之处有：
* 原版在两个GPU上训练两个网络，并在中间某些层设置了两个网络的连接，这里只用一个网络
* 原版有response local normalization，这里没有使用
* 原版的最大池化是overlapping的kernel=3,stide=2的池化，这里直接22
### Idea
可以采用Sequential的写法，这样写forward的时候比较简介，参考自[这里](https://github.com/BIGBALLON/CIFAR-ZOO/blob/master/models/alexnet.py)
```python
class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

## VGG
不加BN的时候loss根本无法下降，后来发现这是因为momentum的原因，把momentum去掉之后就可以缓慢下降了

另一个问题是官方的网络里有AdaptiveAvgPool2d((7,7))的操作，没有这个操作lr在0.01时即便没有momentum
loss也无法下降，这可能是因为下采样过多的原因？？

有趣的现象是训练到后面acc变高但loss也变高

## GoogLeNet
直接用dict来做inception会出现问题，导致dict里面的参数没有register

和VGG一样，没有BN train不动

直接用ImageNet的结构是90.55%，去掉第一个Maxpool是92.74%

GoogLeNet中Inception的四个分路的maxpool+1×1分路的最大池化是3×3 1 stride 1 padding的

### 源码对比结果
* GoogLeNet中的所有pool操作都有一个ceil_mode=True的参数，也就是在特征图上向上取整，在3×3的maxpool中没有使用padding，所以用向上取整来代替
* GoogLeNet的Inception可以写成单独的class，不用像我这样写成字典，看起来很麻烦
* Conv+bn+relu的结构也可以写成单独的类，避免重复写
    ```python
    class BasicConv2d(nn.Module):
    
        def __init__(self, in_channels, out_channels, **kwargs):
            super(BasicConv2d, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            return F.relu(x, inplace=True)
    ```

## CIFAR10
|Model|Acc||
|---|---|---|
|LeNet|75.06%||
|AlexNet|78.06%||
|VGG16|92.64%|可能这就是现在人们浅层网络选择VGG而不选resnet20的原因|
|ResNet20|91.99%|
|GoogLeNet|92.74%|

## image caption
### NIC
preprocess的问题

## faster rcnn
直接用from numpy将图片转成tensor会出错，需要用`torchvision.transforms.functional.to_tensor`

## LSTM
refer to [here](https://www.jianshu.com/p/e091dba8eaaf)
模型本来使用最后一个hidden state，结果不知是梯度的原因还是什么模型在测试集上的准确率一直是67.5，
换成了所有time step hidden state的平均瞬间变成83.04%(sgd lr=0.1)
lr0.01 m0.9 83.68
lr 0.1 decay 0.9/5 82.82
lr 0.1 decay 0.9/5 256 83.71

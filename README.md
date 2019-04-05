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
### 坑
dataset类的transform参数后要打括号，比如transform = ToTensor()，否则会报object...错误

### 笔记
[torch.no_grad](https://pytorch.org/docs/stable/autograd.html?highlight=no_grad#torch.autograd.no_grad)上下文管理器，节约计算资源
CONV-relu-pool
### TODO
训练过程记录入log中 
可视化训练过程 done
保存模型 done
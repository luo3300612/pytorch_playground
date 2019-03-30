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
# import torch
# import torch.nn.functional as F
#
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.tensor([1, 0, 4])
# output = F.nll_loss(F.log_softmax(input), target)
#
# print("nll_loss:")
# print(output)
#
# criterion = torch.nn.CrossEntropyLoss()
# loss =criterion(input,target)
# print("cross entropy:")
# print(loss)

# 初始化参数的时间
# from model import Net
#
# net = Net(4,2)
#
# for parameter in net.parameters():
#     print(parameter)
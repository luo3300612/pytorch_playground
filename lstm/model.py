import torch
import torch.nn as nn
import numpy as np


class myLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(myLSTMCell, self).__init__()
        self.gate = nn.Linear(input_size + hidden_size, 3)
        self.Cell = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, state):
        h, c = state
        x = torch.cat([x, h], dim=1)
        gate = self.gate(x)
        gate_i = torch.sigmoid(gate[:, 0]).view(-1, 1)
        gate_f = torch.sigmoid(gate[:, 1]).view(-1, 1)
        gate_o = torch.sigmoid(gate[:, 2]).view(-1, 1)
        C = torch.tanh(self.Cell(x))
        C = gate_i * C + gate_f * c
        h = gate_o * torch.tanh(C)
        return h, c


class myLSTM(nn.Module):
    def __init__(self, opt):
        super(myLSTM, self).__init__()
        self.input_size = opt.input_size
        self.hidden_size = opt.hidden_size
        self.vocab_size = opt.vocab_size
        self.embedding_dim = opt.embedding_dim
        self.core = myLSTMCell(self.input_size, self.hidden_size)
        self.embed = nn.Embedding(self.vocab_size+1, self.embedding_dim)
        self.max_length = opt.max_length
        self.num_classes = opt.num_classes
        self.output = nn.Linear(self.hidden_size, self.num_classes)

    def init_weight(self, bs):
        weight = next(self.parameters())
        return weight.new_zeros(bs, self.hidden_size), weight.new_zeros(bs, self.hidden_size)

    def forward(self, x):
        state = self.init_weight(x.shape[0])
        outputs = []
        for i in range(self.max_length):
            xt = x[:, i]
            if torch.sum(xt) == 0:
                break
            xt = self.embed(xt)
            state = self.core(xt, state)
            outputs.append(state[0].unsqueeze(2))
        x = self.output(torch.mean(torch.cat(outputs,dim=2),dim=2))
        return x
        # x (bs,seq_len) one line is a seq


if __name__ == '__main__':
    # test cell
    # cell = LSTMCell(10, 10)
    # x = torch.randn(16, 10)
    # c = torch.zeros(16, 10)
    # h = torch.zeros(16, 10)
    # output = cell(x, (h, c))
    # print(output[0].shape)
    # print(output[1].shape)

    # test lstm

    bs = 16
    net = myLSTM(input_size=20, hidden_size=10, vocab_size=15, embedding_dim=20, max_length=10, batch_size=16,
                 num_classes=2)
    x = np.random.randint(1, 15, (bs, 22))
    x = torch.from_numpy(x)
    output = net(x)
    print(output.shape)

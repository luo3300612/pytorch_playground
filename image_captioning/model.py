import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.functional as F


class NIC(nn.Module):
    def __init__(self, opt):
        super(self, NIC).__init__()
        self.vocab_size = opt.vocab_size
        self.input_size = opt.input_size
        self.rnn_size = opt.rnn_size
        self.feat_size = opt.feat_size
        self.num_layers = opt.num_layers

        self.img_embed = nn.Linear(self.feat_size, self.input_size)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_size)
        self.rnn = nn.LSTMCell(self.input_size, self.rnn_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)

    def init_hiddden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.rnn_size),
                weight.new_zeros(self.num_layers, batch_size, self.rnn_size))

    def forward(self, img_feat, seq):
        # seq[:,0] = 0
        batch_size = img_feat.shape[0]
        state = self.init_hiddden(batch_size)
        outputs = []
        for i in range(seq.shape[1]):
            if i == 0:
                x = self.img_embed(img_feat)
            else:
                if seq[:, i].sum() == 0:
                    break
                x = self.embed(seq[:, i])
            output, state = self.rnn(x, state)
            output = F.log_softmax(self.logit(output), dim=1)
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs], dim=1)

from torch.utils.data import DataLoader, Dataset
import torch
import h5py
import json
import random
import argparse


class Data(Dataset):
    def __init__(self, tokens, labels, itow):
        self.tokens = tokens
        self.labels = labels
        self.itow = itow

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]

    def decode(self, tokens):
        return ''.join([self.itow[str(token)] for token in tokens if token != 0])


def get_loader(opt):
    h5_file = h5py.File(opt.data_h5, 'r', driver='core')
    with open(opt.data_json, 'r') as f:
        itow = json.load(f)
    n = h5_file['sentences'].shape[0]
    index = list(range(n))
    random.seed(1)
    random.shuffle(index)
    middle = int(0.7 * n)
    train = sorted(index[:middle])
    test = sorted(index[middle:])
    train_data = Data(h5_file['sentences'][train], h5_file['labels'][train], itow)
    test_data = Data(h5_file['sentences'][test], h5_file['labels'][test], itow)
    print('split dataset, train:{}, test:{}'.format(len(train_data), len(test_data)))
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-h5', type=str,
                        default='/home/luo3300612/Workspace/PycharmWS/pytorch_playground/lstm/data/data.h5')
    parser.add_argument('--data-json', type=str,
                        default='/home/luo3300612/Workspace/PycharmWS/pytorch_playground/lstm/data/data.json')
    parser.add_argument('--batch-size', type=int,
                        default=16)
    opt = parser.parse_args()

    train_loader, test_loader = get_loader(opt)

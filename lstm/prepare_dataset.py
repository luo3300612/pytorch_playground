import pandas as pd
from torch.utils.data import Dataset, DataLoader
import jieba
import argparse
import time
import h5py
import numpy as np
import json

data_path = '/home/luo3300612/Workspace/PycharmWS/pytorch_playground/lstm/data/dataset.csv'
stop_words_path = '/home/luo3300612/Workspace/PycharmWS/pytorch_playground/lstm/data/stop_words'


def read_stop_words(file_path):
    with open(file_path, 'r') as f:
        stop_words = f.readlines()
        stop_words = [word.replace('\n', '') for word in stop_words]
    return stop_words


def build_vocab(data, stop_words, opt):
    word2count = {}
    sent_length = {}

    token_data = {}

    for i in range(data.shape[0]):
        entry = data.iloc[i]
        sentence = entry['review']
        label = entry['label']
        tokens = jieba.cut(sentence)
        tokens = [token for token in tokens if token not in stop_words]
        for token in tokens:
            word2count[token] = word2count.get(token, 0) + 1
        sent_length[len(tokens)] = sent_length.get(len(tokens), 0) + 1
        token_data[i] = {'tokens': tokens, 'label': label}

    cw = sorted([(value, key) for key, value in word2count.items()], reverse=True)
    print('top words and there counts:')
    print('\n'.join(map(str, cw[:20])))

    total_words = sum(word2count.values())
    print('total words:', total_words)
    bad_words = [w for w, n in word2count.items() if n < opt.min_freq]
    vocab = [w for w, n in word2count.items() if n >= opt.min_freq]
    bad_count = sum(word2count[w] for w in bad_words)
    print('number of bad words: {}/{} = {:.2f}%'.format(len(bad_words), total_words,
                                                        len(bad_words) * 100.0 / total_words))
    print('number of words in vocab would be {}'.format(len(vocab), ))
    print('number of UNKs: {}/{} = {:.2f}%'.format(bad_count, total_words, bad_count * 100.0 / total_words))

    print('length, count, percent')
    clip_count = 0
    for length, count in sorted(sent_length.items()):
        print('{} {} {:.2f}%'.format(length, count, count * 100 / data.shape[0]))
        if length > opt.max_length:
            clip_count += count
    print('{} sentences longer than {}, which is {:.2f}%'.format(clip_count, opt.max_length,clip_count * 100 / data.shape[0]))

    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    for i, info in token_data.items():
        info['tokens'] = [token if word2count[token] >= opt.min_freq else 'UNK' for token in info['tokens']]

    return vocab, token_data


def encode_sentences(token_data, wtoi, opt):
    max_length = opt.max_length
    N = len(token_data)
    labels = np.zeros((N,), dtype='int32')
    Li = np.zeros((N, max_length), dtype='int32')
    for i in range(N):
        tokens = token_data[i]['tokens']
        label = token_data[i]['label']
        labels[i] = label
        for j, token in enumerate(tokens):
            if j < max_length:
                Li[i, j] = wtoi[token]
    return Li, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str,
                        default='/home/luo3300612/Workspace/PycharmWS/pytorch_playground/lstm/data/dataset.csv')
    parser.add_argument('--stop-words-path', type=str,
                        default='/home/luo3300612/Workspace/PycharmWS/pytorch_playground/lstm/data/stop_words')
    parser.add_argument('--output', type=str,
                        default='/home/luo3300612/Workspace/PycharmWS/pytorch_playground/lstm/data/')
    parser.add_argument('--min_freq', type=int, default=5, help='word freq less than this number will be remove')
    parser.add_argument('--max_length', type=int, default=20)

    opt = parser.parse_args()

    stop_words = read_stop_words(opt.stop_words_path)
    data = pd.read_csv(opt.data_path)

    vocab, token_data = build_vocab(data, stop_words, opt)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    Li, labels = encode_sentences(token_data, wtoi, opt)
    f_lb = h5py.File(opt.output + 'data.h5', "w")
    f_lb.create_dataset("labels", dtype='int32', data=labels)
    f_lb.create_dataset("sentences", dtype='int32', data=Li)
    f_lb.close()

    with open(opt.output + 'data.json', 'w',encoding='utf8') as f:
        json.dump(itow, f,ensure_ascii=False)

    print('Done')

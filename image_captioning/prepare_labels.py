import torch
import argparse
import json
from random import seed, shuffle
import numpy as np

def build_vocab(imgs, args):
    count_thr = args.word_count_threshold
    counts = {}

    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and there counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: {}/{} = {:.2f}%'.format(len(bad_words), len(counts),
                                                        len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be {}'.format(len(vocab)))
    print('number of UNKs: {}/{}={:.2f}%'.format(bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in rwa data:', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())

    for i in range(max_len + 1):
        print('{}:{},  {}'.format(i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100 / sum_len))

    if bad_count > 0:
        print('inserting the special UNK token')
        vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab

def encode_captions(imgs,args,wtoi):
    max_length = args.max_length
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)

    label_arrays = []
    label_start_ix = np.zeros(N,dtype='uint32')


def main(args):
    imgs = json.load(open(args.input_json, 'r'))
    imgs = imgs['images']

    seed(123)
    vocab = build_vocab(imgs, args)
    itow = {i + 1: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}

    L, label_start_ix, label_end_ix, label_langth = encode_captions(imgs, args, wtoi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare coco label')
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='data.json', help='output json file')
    parser.add_argument('--images_root', default='',
                        help='root location of images')

    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    main(args)

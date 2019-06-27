import torch
import argparse
import json
from random import seed, shuffle
import numpy as np
import h5py
import os
from PIL import Image


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


def encode_captions(imgs, args, wtoi):
    max_length = args.max_length
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s))
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)
    assert L.shape[0] == M, 'length dont match {} != {}'.format(L.shape[0], N)
    assert np.all(label_length > 0), 'error: some caption had no words'

    print('encoded captions to array of size', L.shape)
    return L, label_start_ix, label_end_ix, label_length


def main(args):
    imgs = json.load(open(args.input_json, 'r'))
    imgs = imgs['images']

    seed(123)
    vocab = build_vocab(imgs, args)
    itow = {i + 1: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}

    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, args, wtoi)

    N = len(imgs)
    f_lb = h5py.File(args.output_h5 + '_label.h5', 'w')
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()

    # create output json file
    out = {}
    out['ix_to_word'] = itow
    out['images'] = []
    for i, img in enumerate(imgs):
        jimg = {}
        jimg['split'] = img['split']
        if 'filename' in img: jimg['file_path'] = os.path.join(img['filepath'], img['filename'])
        if 'cocoid' in img: jimg['id'] = img['cocoid']
        if args.images_root != '':
            with Image.open(os.path.join(args.images_root, img['filepath'], img['filename'])) as _img:
                jimg['width'], jimg['height'] = _img.size

        out['images'].append(jimg)

    json.dump(out, open(args.output_json, 'w'))
    print('wrote ', args.output_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare coco label')
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='data.json', help='output json file')
    parser.add_argument('--output_h5', default='data', help='output h5 file')
    parser.add_argument('--images_root', default='',
                        help='root location of images')

    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    print(args)
    main(args)

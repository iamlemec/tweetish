# tweet data

import re
import os
import json
import glob
import html
from collections import Counter, defaultdict
from itertools import chain
import numpy as np

import config

# special tokens
PRIM = ['<android>', '<iphone>', '<other>']
SPEC = ['<unk>', '<end>', '<num>'] + PRIM
DICT = {s: i for (i, s) in enumerate(SPEC)}
ZERO = len(SPEC)

# tweet sources
SRC = {
    'Twitter for Android': '<android>',
    'Twitter for iPhone': '<iphone>'
}

def gen_words(path):
    fid = open(path)
    for line in fid.readlines():
        line = line.strip().lower()
        if len(line) == 0: continue
        yield line.split()

def gen_vocab(raw_data):
    vocab_size = config.vocab_size
    counter = Counter(chain(*raw_data))
    for w in SPEC:
        counter.pop(w, 0)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words = [w for (i, (w, _)) in enumerate(count_pairs) if i < vocab_size - ZERO and w not in SPEC]
    words = SPEC + words
    return dict(zip(words, range(len(words))))

def load_data(data_path):
    raw_data = list(gen_words(data_path))
    vocab = gen_vocab(raw_data)
    print('vocab = %d' % len(vocab))

    iunk = DICT['<unk>']
    iend = DICT['<end>']
    num_steps = config.num_steps

    pad_row = lambda row: row[:num_steps] + [iend]*max(0,num_steps-len(row))
    wgt_row = lambda row: [1]*min(num_steps,len(row)+1) + [0]*max(0,num_steps-len(row)-1)

    data_ids = [[vocab.get(word, iunk) for word in row] for row in raw_data]
    output_data = [pad_row(row[1:]) for row in data_ids]
    input_data = [pad_row(row) for row in data_ids]
    weights = [wgt_row(row) for row in data_ids]

    return (vocab, input_data, output_data, weights)

class Feeder(object):
    def __init__(self, data_path):
        self.n_samples = len(self.input_data)
        self.vocab, self.input_data, self.output_data, self.weights = load_data(data_path)
        self.ivocab = defaultdict(lambda: '<unk>', {v: k for (k, v) in self.vocab.items()})

        self.n_train = (1-config.valid_frac)*self.n_samples
        self.b_train = self.n_train // config.batch_size
        self.i_train = 0

        self.n_valid = config.valid_frac*self.n_samples
        self.b_valid = self.n_valid // config.batch_size
        self.i_valid = 0

        print('b_train = %d' % b_train)
        print('b_valid = %d' % b_valid)

    def get_train(self):
        batch_size = config.batch_size
        row = self.i_train*batch_size
        if row + batch_size > self.n_train:
            self.i_train = 0
            row = 0
        self.i_train += 1
        return (self.input_data[row:row+batch_size],
                self.output_data[row:row+batch_size],
                self.weights[row:row+batch_size])

    def get_valid(self):
        batch_size = config.batch_size
        row = self.n_train + self.i_valid*batch_size
        if row + batch_size > self.n_samples:
            self.i_valid = 0
            row = self.n_train
        self.i_valid += 1
        return (self.input_data[row:row+batch_size],
                self.output_data[row:row+batch_size],
                self.weights[row:row+batch_size])

def gen_tweets(js, retweets=False, replies=False, number=True, device=True):
    for tw in js:
        source = tw['source']
        text = tw['text'].lower().strip()

        if not retweets and (tw['is_retweet'] or text.startswith('"@') or text.startswith('rt')):
            continue
        if not replies and tw['in_reply_to_user_id_str'] is not None:
            continue

        stok = SRC.get(source, '<other>')

        # odd subs
        text = re.sub(r'’', '\'', text)
        text = re.sub(r'—', '-', text)

        # urls
        text = re.sub(r'\bhttps?://[\S]*\b', r' ', text)
        text = html.unescape(text)

        # acronyms
        text = re.sub(r'\.{2,}', r'.', text)
        text = re.sub(r'(\S)\.(\S)\.', r'\1\2', text)
        text = re.sub(r'(\S)\.(\S)\.(\S)\.', r'\1\2\3', text)

        # control chars
        text = re.sub(r'([!\.&,])', r' \1 ', text)
        text = re.sub(r'[^ a-zA-Z0-9#@!\.\'&]', r' ', text)

        # numbers
        if number:
            text = re.sub(r'\b(\d\S*|[^#]\S*\d\S*)\b', r' <num> ', text)

        # clean up
        text = re.sub(r' {2,}', r' ', text)
        text = text.lower().strip()

        # device type
        if device:
            text = stok + ' ' + text

        # combine
        yield text

def parse_data(inputs, out_path=None, retweets=False, replies=False, number=True, device=True):
    tweets = []
    if type(inputs) is not list:
        inputs = glob.glob(inputs)
    for fpath in inputs:
        js = json.load(open(fpath))
        tweets += gen_tweets(js, retweets=retweets, replies=replies, number=number, device=device)

    if out_path is None:
        return tweets
    else:
        fid = open(out_path, 'w')
        fid.write('\n'.join(tweets))


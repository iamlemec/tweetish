# tweet data

import re
import os
import json
import glob
import html
from collections import Counter
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

    num_steps = config.num_steps
    pad_row = lambda row: row[:num_steps] + [DICT['<end>']]*max(0,num_steps-len(row))
    wgt_row = lambda row: [1]*min(num_steps,len(row)+1) + [0]*max(0,num_steps-len(row)-1)

    data_ids = [[vocab.get(word, DICT['<unk>']) for word in row] for row in raw_data]
    output_data = [pad_row(row[1:]) for row in data_ids]
    input_data = [pad_row(row) for row in output_data]
    weights = [wgt_row(row) for row in data_ids]

    return (vocab, input_data, output_data, weights)

class Feeder(object):
    def __init__(self, data_path, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.vocab, self.input_data, self.output_data, self.weights = load_data(data_path)
        self.ivocab = {v: k for (k, v) in self.vocab.items()}
        self.n_samples = n_samples = len(self.input_data)
        self.epoch_size = ((n_samples // batch_size) - 1) // num_steps
        self.batch_num = 0
        print('epoch_size = %d' % self.epoch_size)

    def get_batch(self):
        batch_size = self.batch_size
        row = self.batch_num*batch_size
        if row + batch_size > len(self.input_data):
            self.batch_num = 0
            row = 0
        return (self.input_data[row:row+batch_size],
                self.output_data[row:row+batch_size],
                self.weights[row:row+batch_size])

def gen_tweets(js):
    for tw in js:
        source = tw['source']
        text = tw['text'].lower().strip()

        if not config.retweets and (tw['is_retweet'] or text.startswith('"@') or text.startswith('rt')):
            continue
        if not config.replies and tw['in_reply_to_user_id_str'] is not None:
            continue

        stok = SRC.get(source, '<other>')

        text = re.sub(r'\bhttps?://[\S]*\b', r' ', text)
        text = html.unescape(text)
        text = re.sub(r'\.{2,}', r'.', text)
        text = re.sub(r'(\S)\.(\S)\.', r'\1\2', text)
        text = re.sub(r'(\S)\.(\S)\.(\S)\.', r'\1\2\3', text)
        text = re.sub(r'([!\.&,])', r' \1 ', text)
        text = re.sub(r'[^ a-z0-9#@!\.&]', r'', text)
        text = re.sub(r'\b\d+(p|pm|a|am)\b', r'<time>', text)
        text = re.sub(r'\b\d{1,3}\b', r'<num>', text)
        text = re.sub(r'\b(\D+\d\D*|\D*\d\D+)\b', r'', text)
        text = re.sub(r' {2,}', r' ', text)
        text = text.lower().strip()

        yield stok + ' ' + text

def parse_data(inp_path, out_path=None):
    tweets = []
    for fpath in glob.glob(inp_path):
        js = json.load(open(fpath))
        tweets += gen_tweets(js)

    if out_path is None:
        return tweets
    else:
        fid = open(out_path, 'w')
        fid.write('\n'.join(tweets))


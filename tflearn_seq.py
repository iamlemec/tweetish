from __future__ import absolute_import, division, print_function

import os, sys, argparse
import urllib

import tflearn
from tflearn.data_utils import *

parser = argparse.ArgumentParser(description=
    'Pass a text file to generate LSTM output')

parser.add_argument('filename')
parser.add_argument('-t','--temp', type=float, default=None, help=
    'Defaults to displaying multiple temperature outputs which is suggested.' +
    ' If temp is specified, a value of 0.0 to 2.0 is recommended.' +
    ' Temperature is the novelty or' +
    ' riskiness of the generated output.  A value closer to 0 will result' +
    ' in output closer to the input, so higher is riskier.'
)
parser.add_argument('-l','--length', type=int, default=25, help=
    'Optional length of text sequences to analyze.  Defaults to 25.'
)

args = vars(parser.parse_args())

size = 256
drop = 0.5
temp = args['temp']
maxlen = args['length']
fname = args['filename']
model_name = os.path.split(fname)[1].split('.')[0]  # create model name from textfile input

X, Y, char_idx = \
    textfile_to_semi_redundant_sequences(fname, seq_maxlen=maxlen, redun_step=3)

g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, size, return_seq=True)
g = tflearn.dropout(g, drop)
g = tflearn.lstm(g, size, return_seq=True)
g = tflearn.dropout(g, drop)
g = tflearn.lstm(g, size)
g = tflearn.dropout(g, drop)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_'+ model_name)

for i in range(50):
    seed = random_sequence_from_textfile(fname, maxlen)
    m.fit(X, Y, validation_set=0.1, batch_size=128,
          n_epoch=1, run_id=model_name)
    print("-- TESTING...")
    if temp is not None:
        print("-- Test with temperature of %s --" % temp)
        print(m.generate(600, temperature=temp, seq_seed=seed))
    else:
        print("-- Test with temperature of 1.0 --")
        print(m.generate(600, temperature=1.0, seq_seed=seed))
        print("-- Test with temperature of 0.5 --")
        print(m.generate(600, temperature=0.5, seq_seed=seed))

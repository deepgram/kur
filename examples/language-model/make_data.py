"""
Copyright 2016 Deepgram

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from vocab import *
import json
import os


if not os.path.exists('./data/'):
    os.mkdir('./data/')

def one_hot(v, ndim):
    v_one_hot = np.zeros(
        (len(v), ndim,)
    )
    for i in range(len(v)):
        v_one_hot[i][v[i]] = 1.0
    return v_one_hot

x = []
y = []

all_chars = []
for book in [
    'pride_and_prejudice.txt',
    'shakespeare.txt'
]:
    with open('books/%s' % book, 'r') as infile:
        chars = [
            c for c in ' '.join(infile.read().lower().split())
            if c in set(vocab)
        ]
        all_chars += [' ']
        all_chars += chars

all_chars = list(' '.join(''.join(all_chars).split()))
num_chars = len(all_chars)
with open('cleaned.txt', 'w') as outfile:
    outfile.write(''.join(all_chars))


x, y = [], []

data_portions = [
    ('train', 0.8),
    ('validate', 0.05),
    ('test', 0.05),
    ('evaluate', 0.05),
]

dev = False
if dev:
    # shrink data to make things go faster
    for i in range(len(data_portions)):
        data_portions[i] = (
            data_portions[i][0],
            data_portions[i][1] * 0.1
        )

max_i = sum([
    int(round(len(all_chars) * fraction))
    for name, fraction in data_portions
]) - seq_len

for i in range(max_i):

    in_char_seq = all_chars[i: i + seq_len]

    # one hot representation
    sample_x = np.zeros((len(in_char_seq), n_vocab,))
    for j, c in enumerate(in_char_seq):
        sample_x[j][char_to_int[c]] = 1
    x.append(sample_x)

    sample_y = np.zeros(n_vocab)
    sample_y[char_to_int[all_chars[i + seq_len]]] = 1
    y.append(sample_y)

x, y = np.array(x).astype('int32'), np.array(y).astype('int32')

start_i = 0
for name, fraction in data_portions:
    end_i = start_i + int(round(len(x) * fraction))
    print(start_i, end_i)
    x0 = x[start_i: end_i]
    y0 = y[start_i: end_i]

    print('dims:')
    print(x0.shape)
    print(y0.shape)

    start_i = end_i

    with open('data/%s.jsonl' % name, 'w') as outfile:
        for sample_x, sample_y in zip(x0, y0):
            outfile.write(json.dumps({
                'in_seq': sample_x.tolist(),
                'out_char': sample_y.tolist()
            }))
            outfile.write('\n')

    del x0, y0

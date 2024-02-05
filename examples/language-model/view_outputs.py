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
import pickle
import sys
import view_data
from vocab import *


if len(sys.argv) < 2:
    pickle_fname = 'model/output.pkl'
else:
    pickle_fname = sys.argv[1]

with open(pickle_fname, 'rb') as infile:
    prediction_data = pickle.load(infile)

data = view_data.get_data('evaluate')

batch_size = len(prediction_data['truth']['out_char'])

for j in range(10):
    predicted_char = int_to_char[np.argmax(prediction_data['result']['out_char'][j])]
    correct_char = int_to_char[np.argmax(data['out_char'][j])]
    print(
        '"%s" --> "%s"' % (
            ''.join([
                int_to_char[np.argmax(_)]
                for _ in data['in_seq'][j]
            ]),
            predicted_char
        )
    )
    if predicted_char == correct_char:
        print((' ' * (seq_len + 5)) + 'CORRECT')
    else:
        print((' ' * (seq_len + 5)) + 'INCORRECT (%s)' % correct_char)

accuracy = sum(int(np.argmax(prediction_data['result']['out_char'][i]) == np.argmax(prediction_data['truth']['out_char'][i])) for i in range(batch_size)) / float(len(prediction_data['truth']['out_char']))

print('accuracy = %s' % accuracy)
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
import json
from vocab import *


# this script shows sections of size `window` from the ends
# of each data file. its output looks like this:
# peek at train:
# ------
# "the project gutenberg ebook of" --> " "
# "he project gutenberg ebook of " --> "p"
# "e project gutenberg ebook of p" --> "r"
# ------
# "o particular resentment by his" --> " "
# " particular resentment by his " --> "h"
# "particular resentment by his h" --> "a"


# peek at validate:
# ------
# "articular resentment by his ha" --> "v"
# "rticular resentment by his hav" --> "i"
# "ticular resentment by his havi" --> "n"
# ------
# "ingley for a kingdom upon my h" --> "o"
# "ngley for a kingdom upon my ho" --> "n"
# "gley for a kingdom upon my hon" --> "o"


# peek at test:
# ------
# "ley for a kingdom upon my hono" --> "u"
# "ey for a kingdom upon my honou" --> "r"
# "y for a kingdom upon my honour" --> " "
# ------
# "sting your time with me. mr. b" --> "i"
# "ting your time with me. mr. bi" --> "n"
# "ing your time with me. mr. bin" --> "g"


# peek at evaluate:
# ------
# "ng your time with me. mr. bing" --> "l"
# "g your time with me. mr. bingl" --> "e"
# " your time with me. mr. bingle" --> "y"
# ------
# "they had yet learnt to care fo" --> "r"
# "hey had yet learnt to care for" --> " "
# "ey had yet learnt to care for " --> "a"


window = 3

def get_data(name):
    with open('data/%s.jsonl' % name, 'r') as infile:
        data = {}

        data = {
            key: [value]
            for key, value in json.loads(next(infile)).items()
        }
        for line in infile:
            line = line.strip()
            if not line:
                continue
            for key, value in json.loads(line).items():
                data[key].append(value)

        for key in data:
            data[key] = np.array(data[key])

    return data


def view_data(name):
    data = get_data(name)

    print('\n\npeek at %s:' % name)
    for j_range in (range(window), range(len(data['in_seq']) - window, len(data['in_seq'])),):
        print('------')
        for j in j_range:
            print(
                '"%s" --> "%s"' % (
                    ''.join([
                        int_to_char[np.argmax(_)]
                        for _ in data['in_seq'][j]
                    ]),
                    int_to_char[np.argmax(data['out_char'][j])]
                )
            )

if __name__ == '__main__':
    for name in ('train', 'validate', 'test', 'evaluate',):
        view_data(name)

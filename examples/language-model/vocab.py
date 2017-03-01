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

# length of the context sequence
seq_len = 30

lowercase_letters = [
    chr(97 + i) for i in range(26)
]
symbols = [' ', '"', '\'', '.']

# these are the characters we allow in our data
vocab = lowercase_letters + symbols

# we can convert between our vocab and integers
char_to_int = dict(
    (c, i) for i, c in enumerate(vocab)
)
int_to_char = dict(enumerate(vocab))

n_vocab = len(vocab)

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

import json
from . import Supplier
from ..sources import TextSource, TextLength, RawText


###############################################################################
class TextSupplier(Supplier):
    """ A supplier which parses out data from a JSON-Lines file (*.jsonl)
        the keys from the first JSON line object determine the source names, and
        all subsequent lines should have the same keys. The values should be strings
        which will be represented as one-hot encoding under a vocab.
    """

    ###########################################################################
    @classmethod
    def get_name(cls):
        """ Returns the name of the supplier.
        """
        return 'text'

    ###########################################################################
    def __init__(self, path, seq_len, vocabs, padding=None, pad_with=None, *args, **kwargs):
        """ Creates a new JSONL text dictionary supplier.

            # Arguments

            path: str. Filename of JSONL file (*.jsonl) to load.
            vocabs: dict. vocabs[key] = list of vocab items for this source
            padding: dict (optional0. padding[key] = 'left' or 'right' for 0 padding
        """
        super().__init__(*args, **kwargs)

        self.path = path

        with open(self.path, 'r') as infile:
            keys = json.loads(next(infile)).keys()

        if type(seq_len) == dict:
            self.seq_len = seq_len
        else:
            self.seq_len = {
                k: seq_len
                for k in keys
            }

        self.vocabs = vocabs

        if padding is None:
            self.padding = {
                k: 'right'
                for k in keys
            }
        else:
            self.padding = padding

        if pad_with is None:
            self.pad_with = {
                k: None
                for k in keys
            }
        else:
            self.pad_with = pad_with

        self.vocabs = {
            k: list(map(str, self.vocabs[k]))
            for k in self.vocabs
        }

        self.sources = None

        with open(self.path, 'r') as infile:
            self.num_entries = sum(1 for _ in infile if _.strip())

        self.sources = {}
        for k in keys:
            self.sources['raw_' + k] = RawText(
                self.path,
                k,
                self.num_entries
            )
            self.sources[k] = TextSource(
                'raw_' + k,
                self.vocabs[k],
                self.num_entries,
                self.seq_len[k],
                padding=self.padding[k],
                pad_with=self.pad_with[k]
            )
            self.sources[k + '_length'] = TextLength(
                'raw_' + k,
                self.num_entries
            )


    ###########################################################################
    def get_sources(self, sources=None):
        """ Returns all sources from this provider.
        """

        if sources is None:
            with open(self.path, 'r') as infile:
                keys = list(json.loads(next(infile)).keys())
                sources = [
                    k for k in keys
                ] + [
                    'raw_' + k for k in keys
                ] + [
                    k + '_length' for k in keys
                ]

        elif not isinstance(sources, (list, tuple)):
            sources = [sources]

        for source in sources:
            if source not in self.sources:
                raise KeyError(
                    'Invalid data key: {}. Valid keys are: {}'.format(
                        source, ', '.join(str(k) for k in self.sources.keys())
                ))

        return {k : self.sources[k] for k in sources}

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

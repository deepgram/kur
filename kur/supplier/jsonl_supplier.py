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
from ..sources import JSONLSource

###############################################################################
class JSONLSupplier(Supplier):
    """ A supplier which parses out data from a JSON-Lines file (*.jsonl)
        the keys from the first JSON line object determine the source names, and
        all subsequent lines should have the same keys. Because JSON supports arrays,
        the JSONL Supplier is good for arbitrary tensor shapes, and scales to lots of
        data.
    """

    ###########################################################################
    @classmethod
    def get_name(cls):
        """ Returns the name of the supplier.
        """
        return 'jsonl'

    ###########################################################################
    def __init__(self, source, *args, **kwargs):
        """ Creates a new JSONL dictionary supplier.

            # Arguments

            source: str. Filename of JSONL file (*.jsonl) to load.
        """
        super().__init__(*args, **kwargs)

        self.source = source
        self.data = None
        with open(self.source, 'r') as infile:
            keys = json.loads(next(infile)).keys()

        with open(self.source, 'r') as infile:
            self.num_entries = sum(1 for _ in infile if _.strip())

        self.data = {
            k: JSONLSource(self.source, k, self.num_entries)
            for k in keys
        }

    ###########################################################################
    def get_sources(self, sources=None):
        """ Returns all sources from this provider.
        """

        if sources is None:
            with open(self.source, 'r') as infile:
                sources = list(json.loads(next(infile)).keys())

        elif not isinstance(sources, (list, tuple)):
            sources = [sources]

        for source in sources:
            if source not in self.data:
                raise KeyError(
                    'Invalid data key: {}. Valid keys are: {}'.format(
                        source, ', '.join(str(k) for k in self.data.keys())
                ))

        return {k : self.data[k] for k in sources}

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

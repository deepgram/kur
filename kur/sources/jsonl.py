import linecache
import numpy
import json

from ..sources import ChunkSource


###############################################################################
class JSONLSource(ChunkSource):
    """ Data source for tensors stored in JSONL format
    """

    ###########################################################################
    def __init__(self, source, key, num_entries, *args, **kwargs):
        """ Creates a new JSONL source for file named `source`.
        """

        super().__init__(*args, **kwargs)

        self.source = source
        self.num_entries = num_entries
        self.key = key
        self.indices = numpy.arange(len(self))

    ###########################################################################
    def __iter__(self):
        """ Return an iterator to the data. Get the value (tensor) for self.key
        from each object and yield batches of these tensors
        """
        start = 0
        while start < self.num_entries:
            end = min(self.num_entries, start + self.chunk_size)

            # linecache line numbering starts at 1
            batch = [
                json.loads(linecache.getline(self.source, i + 1).strip())[self.key]
                for i in self.indices[start:end]
            ]

            yield batch
            start = end

    ###########################################################################
    def __len__(self):
        """ Returns the total number of entries that this source can return, if
            known.
        """
        return self.num_entries

    ###########################################################################
    def shape(self):
        """ Return the shape of the tensor (excluding batch size) returned by
            this data source.
        """
        return numpy.array(json.loads(linecache.getline(self.source, 0 + 1))[self.key]).shape

    ###########################################################################
    def can_shuffle(self):
        """ This source can be shuffled.
        """
        return True

    ###########################################################################
    def shuffle(self, indices):
        """ Applies a permutation to the data.
        """
        if len(indices) > len(self):
            raise ValueError('Shuffleable was asked to apply permutation, but '
                'the permutation is longer than the length of the data set.')
        self.indices[:len(indices)] = self.indices[:len(indices)][indices]

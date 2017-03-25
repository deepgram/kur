import linecache
import numpy
import json

from ..sources import ChunkSource, DerivedSource


###############################################################################
class TextLength(DerivedSource):
    """ Data source for audio lengths.
    """
    def __init__(self, source, num_entries):
        super().__init__()
        self.source = source
        self.num_entries = num_entries
    def derive(self, inputs):
        text_chunks, = inputs
        return numpy.array([[len(x)] for x in text_chunks], dtype='int32')
    def shape(self):
        return (1,)
    def requires(self):
        return (self.source, )
    def __len__(self):
        return self.num_entries

###############################################################################
class TextSource(DerivedSource):
    """ Data source for neat (non-ragged) one-hot represented text arrays.
    """
    def __init__(self, source, vocab, num_entries, seq_len, padding='right', pad_with=None):
        super().__init__()
        self.num_entries = num_entries
        self.source = source
        self.seq_len = seq_len
        self.padding = padding
        self.pad_with = pad_with
        self.vocab = vocab
        self.char_to_int = {
            c: i
            for i, c in enumerate(self.vocab)
        }

    def _encode(self, char_seq):
        output = numpy.zeros(shape=(len(char_seq), len(self.vocab),))
        for i in range(len(char_seq)):
            output[i][self.char_to_int[char_seq[i]]] = 1
        assert output.shape[0] == len(char_seq)
        return output

    def derive(self, inputs):
        text_chunk, = inputs

        output = numpy.zeros(
            shape=(
                len(text_chunk),
                self.seq_len,
                len(self.vocab),
            ),
            dtype='int32'
        )

        for i, char_seq in enumerate(text_chunk):
            char_seq = list(char_seq)
            if self.padding == 'right':
                if self.pad_with is not None:
                    char_seq = char_seq + (self.seq_len - len(char_seq)) * [self.pad_with]
                encoded_char_seq = self._encode(char_seq)

                assert len(encoded_char_seq) == len(char_seq)

                for j in range(len(encoded_char_seq)):
                    output[i][j] = encoded_char_seq[j]

            elif self.padding == 'left':
                if self.pad_with is not None:
                    char_seq = (self.seq_len - len(char_seq)) * [self.pad_with] + char_seq
                encoded_char_seq = self._encode(char_seq)

                assert len(encoded_char_seq) == len(char_seq)

                for j in range(len(encoded_char_seq)):
                    output[i][-len(char_seq) + j] = encoded_char_seq[j]

            else:
                raise ValueError('Padding must be left or right, not %s' % padding)
        return output

    def shape(self):
        """ Return the shape of the tensor (excluding batch size) returned by
            this data source.
        """
        return (self.seq_len, len(self.vocab),)

    def requires(self):
        return (self.source, )

    def __len__(self):
        return self.num_entries

###############################################################################
class RawText(ChunkSource):
    """ Data source for text stored in JSONL format like:
    ['a', 'p', 'p', 'l', 'e', ' ', 'p', 'i', 'e']
    """

    ###########################################################################
    @classmethod
    def default_chunk_size(cls):
        """ Returns the default chunk size for this source.
        """
        return 256

    ########################################################################### 
    def shape(self):
        return (None,)

    ###########################################################################
    def __init__(self, source, key, num_entries, *args, **kwargs):
        """ Creates a new Text source for file named `source`.
        """

        super().__init__(*args, **kwargs)

        self.source = source
        self.num_entries = num_entries
        self.key = key
        self.indices = numpy.arange(len(self))

    ###########################################################################
    def __iter__(self):
        """ Return an iterator to the data. Yield the value for self.key
        from each object
        """
        start = 0
        while start < self.num_entries:
            end = min(self.num_entries, start + self.chunk_size)

            # linecache line numbering starts at 1
            batch = [
                json.loads(
                    linecache.getline(
                        self.source,
                        i + 1
                    ).strip()
                )[self.key]
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

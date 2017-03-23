"""
Copyright 2017 Deepgram

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

import logging
import numpy
from . import EvaluationHook#, TrainingHook

logger = logging.getLogger(__name__)


###############################################################################
class TextHook(EvaluationHook):#, TrainingHook):
    """ Post-evaluation hook for text data.
    """

    ###########################################################################
    @classmethod
    def get_name(cls):
        """ Returns the name of the evaluation hook.
        """
        return 'text'

    ###########################################################################
    @staticmethod
    def argmax_decode(output, rev_vocab):
        """ output = matrix: timesteps * characters
        """
        x = numpy.argmax(output, axis=1)
        tokens = []
        for c in x:
            tokens.append(c)
        return ''.join([rev_vocab[i] for i in tokens])

    ###########################################################################
    def __init__(self, **kwargs):
        """ Creates a new text hook.
        """

        super().__init__(**kwargs)

    ###########################################################################
    def apply(self, current, original, model=None):
        """ Applies the hook to the data.
        """

        _raw_input = {
            k: ''.join(original[1]['raw_' + k][0])
            for k in model.inputs.keys()
        }

        data, truth = current

        raw_truth = {
            k: ''.join(truth['raw_' + k][0])
            for k in model.outputs.keys()
        }

        vocabs = {
            k: model.provider.sources[model.provider.keys.index(k)].vocab
            for k in model.outputs.keys()
        }

        prediction = {
            k: self.argmax_decode(
                data[k][0],
                vocabs[k]
            )
            for k in model.outputs.keys()
        }

        print('Input:\n{}'.format('\n'.join([
            '%s: "%s"' % (k, v)
            for k, v in _raw_input.items()
        ])), flush=True)
        print('Prediction:\n{}'.format('\n'.join([
            '%s: "%s"' % (k, v)
            for k, v in prediction.items()
        ])), flush=True)
        print('Truth:\n{}'.format('\n'.join([
            '%s: "%s"' % (k, v)
            for k, v in raw_truth.items()
        ])), flush=True)
        return (prediction, raw_truth)

    # ###########################################################################
    # def notify(self, status, log=None, info=None):
    #     """ do not notify.
    #     """
    #     pass


#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

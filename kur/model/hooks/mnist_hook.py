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

import logging
import numpy
from . import EvaluationHook

logger = logging.getLogger(__name__)

###############################################################################
class MnistHook(EvaluationHook):
	""" Post-evaluation hook for MNIST data, which prints summary statistics
		specific to the MNIST data set.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the evaluation hook.
		"""
		return 'mnist'

	###########################################################################
	def apply(self, current, original, model=None):
		""" Applies the hook to the data.
		"""
		data, truth = original
		if truth is None:
			logger.warning('MNIST hook only works with ground-truth data. We '
				'will skip this hook.')
			return current

		stats = {
			i : {'correct' : 0, 'total' : 0}
			for i in range(10)
		}

		if len(data) == 1:
			for key in data:
				pass
		elif len(data) > 1:
			if 'labels' in data:
				key = 'labels'
			else:
				logger.error('Cannot parse this data in the MNIST hook. Too '
					'many keys, none of which is "labels".')
				return current
		else:
			logger.error('MNIST hook cannot process this output: it has no '
				'keys.')
			return current

		if key not in truth:
			logger.error('No ground truth data present for MNIST hook.')
			return current

		if len(truth[key]) != len(data[key]):
			logger.warning('There appears to be a different amount of ground '
				'truth information than model output.')
			return current

		decoded_estimate = numpy.argmax(data[key], axis=1)
		decoded_truth = numpy.argmax(truth[key], axis=1)

		for estimate, truth in zip(decoded_estimate, decoded_truth):
			if truth not in stats:
				logger.warning('Out-of-range truth information: %d', truth)
			else:
				stats[truth]['total'] += 1
				if truth == estimate:
					stats[truth]['correct'] += 1

		print(('{: <10s}' * 4).format(
			'LABEL', 'CORRECT', 'TOTAL', 'ACCURACY'
		))
		for i in range(10):
			print(('{: <10d}'*3 + '{: >5.1f}%').format(
				i, stats[i]['correct'], stats[i]['total'],
				stats[i]['correct'] * 100. / stats[i]['total'] \
					if stats[i]['total'] else 0
			))

		correct = sum(x['correct'] for x in stats.values())
		total = sum(x['total'] for x in stats.values())
		print(('{: <10s}' + '{: <10d}'*2 + '{: >5.1f}%').format(
			'ALL', correct, total, correct * 100. / total if total else 0
		))

		return current

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

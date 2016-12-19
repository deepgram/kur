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
from . import ShuffleProvider

logger = logging.getLogger(__name__)

###############################################################################
class BatchProvider(ShuffleProvider): # pylint: disable=too-few-public-methods
	""" A provider which tries to return the same number of samples at every
		batch.

		All batches except for the last batch are guaranteed to be constant.
		The last batch might be different if there are not enough samples
		available to populate a complete batch.
	"""

	DEFAULT_BATCH_SIZE = 32

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the provider.
		"""
		return 'batch_provider'

	###########################################################################
	def __init__(self, batch_size=None, num_batches=None, *args, **kwargs):
		""" Creates a new batch provider.

			# Arguments

			batch_size: int. The number of batches to return each iteration.
		"""
		super().__init__(*args, **kwargs)
		self.batch_size = batch_size or BatchProvider.DEFAULT_BATCH_SIZE
		logger.info('Batch size set to: %d', self.batch_size)

		self.num_batches = num_batches

		if self.num_batches is not None:
			logger.info('Maximum number of batches set to: %d',
				self.num_batches)
			if self.entries > 0:
				self.entries = min(
					self.entries,
					self.num_batches * self.batch_size
				)

	###########################################################################
	def add_source(self, source, name=None):
		""" Adds a new data source to an existing provider.
		"""
		super().add_source(source, name=name)
		if self.num_batches is not None:
			if self.entries > 0:
				self.entries = min(
					self.entries,
					self.num_batches * self.batch_size
				)

	###########################################################################
	def __iter__(self):
		""" Retrieves the next batch of data.

			# Return value

			A tensor of shape `(X, ) + self.shape()`, where `X` is the number
			of entries provided by this data source.
		"""

		# Should always be the first call in __iter__
		super().pre_iter()

		iterators = [iter(source) for source in self.sources]
		queues = [next(it) for it in iterators]
		sentinel = object()
		proceed = True
		batches_produced = 0

		while iterators and proceed:

			result = [None for it in iterators]

			# Go over each data source.
			for i, it in enumerate(iterators):

				# Get enough data out of each one.
				while len(queues[i]) < self.batch_size:

					# Get the next batch. If there isn't any data left, flag
					# it.  After all, we may have collected at least some data
					# during the `while` loop, and we should return that data.
					x = next(it, sentinel)
					if x is sentinel:
						proceed = False
						break

					# Add the data to the queue.
					queues[i] = numpy.concatenate([queues[i], x])

				# Get the data ready.
				result[i] = queues[i][:self.batch_size]
				queues[i] = queues[i][self.batch_size:]

			lens = {len(q) for q in result}
			if len(lens) == 1:
				if lens.pop():
					yield self.wrap(result)

					batches_produced += 1
					if self.num_batches is not None:
						if batches_produced >= self.num_batches:
							break
				else:
					break
			else:
				if proceed:
					raise ValueError('Managed to accumulate unequal-length '
						'batches, but we were still told to continue '
						'iterating. This is a bug.')

				smallest = min(lens)
				if smallest == 0:
					break
				result = [x[:smallest] for x in result]
				yield self.wrap(result)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

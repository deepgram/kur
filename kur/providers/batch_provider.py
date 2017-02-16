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
from ..sources import ChunkSource
from ..utils import neighbor_sort

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
	def __init__(self, batch_size=None, num_batches=None,
		force_batch_size=False, neighborhood_sort=None, neighborhood_size=None,
		neighborhood_growth=None, *args, **kwargs):
		""" Creates a new batch provider.

			# Arguments

			batch_size: int. The number of batches to return each iteration.
		"""
		super().__init__(*args, **kwargs)

		self.batch_size = batch_size or BatchProvider.DEFAULT_BATCH_SIZE
		logger.debug('Batch size set to: %d', self.batch_size)

		self.num_batches = num_batches

		self.force = force_batch_size
		if self.force:
			logger.debug('Batch provider will force batches of exactly %d '
				'samples.', self.batch_size)

		if neighborhood_sort:
			if self.keys is None:
				raise ValueError('Cannot use "neighborhood_sort" with unnamed '
					'sources.')
			try:
				neighborhood_data = self.sources[self.keys.index(neighborhood_sort)]
			except ValueError:
				raise ValueError('Could not find the "neighborhood_sort" key '
					'"{}" in list of available sources: {}'
					.format(neighborhood_sort, ', '.join(self.keys)))

			if len(neighborhood_data) <= 0:
				raise ValueError('Data sorting requires a finite source.')
		else:
			neighborhood_data = None

		self.neighborhood_data = neighborhood_data
		self.neighborhood_key = neighborhood_sort
		self.neighborhood_size = neighborhood_size
		self.neighborhood_growth = neighborhood_growth

	###########################################################################
	@property
	def num_batches(self):
		""" Returns the number of batches.
		"""
		return self._num_batches

	###########################################################################
	@num_batches.setter
	def num_batches(self, value):
		""" Sets the number of batches this provider should use per epoch.
		"""
		self._num_batches = value # pylint: disable=attribute-defined-outside-init
		if self.num_batches is not None:
			logger.debug('Maximum number of batches set to: %d',
				self.num_batches)
			if self.entries > 0:
				self.entries = min( # pylint: disable=attribute-defined-outside-init
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
				self.entries = min( # pylint: disable=attribute-defined-outside-init
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
		self.pre_iter()

		for source in self.sources:
			if isinstance(source, ChunkSource):
				if source.requested_chunk_size is ChunkSource.USE_BATCH_SIZE:
					source.set_chunk_size(self.batch_size)

		logger.debug('Preparing next batch of data...')

		iterators = [iter(source) for source in self.sources]
		ordering = self.order_sources()
		dependencies = self.source_dependencies()
		queues = [[] for it in iterators]
		sentinel = object()
		proceed = True
		batches_produced = 0

		for it, source in zip(iterators, self.sources):
			if source.is_derived():
				next(it)

		while iterators and proceed:

			if batches_produced:
				logger.debug('Preparing next batch of data...')

			result = [None for it in iterators]

			# Go over each data source.
			for i in ordering:
				it = iterators[i]
				source = self.sources[i]
				depends = dependencies[i]

				if source.is_derived():
					requirements = [result[k] for k in depends]
					if any(x is None for x in requirements):
						raise ValueError('One of the dependent data sources '
							'has not been calculated yet. This is a bug.')
					try:
						result[i] = it.send(requirements)
						next(it)
					except StopIteration:
						proceed = False

				else:
					# Get enough data out of each one.
					while len(queues[i]) < self.batch_size:

						# Get the next batch. If there isn't any data left,
						# flag it.  After all, we may have collected at least
						# some data during the `while` loop, and we should
						# return that data.
						x = next(it, sentinel)
						if x is sentinel:
							proceed = False
							break

						# Add the data to the queue.
						if not queues[i]:
							queues[i] = x
						else:
							queues[i].extend(x)

					# Get the data ready.
					result[i] = numpy.array(queues[i][:self.batch_size])
					queues[i] = queues[i][self.batch_size:]

					if not result[i].shape[0]:
						logger.debug('An original source has no data.')
						return

			logger.debug('Next batch of data has been prepared.')

			lens = {len(q) for q in result}
			if len(lens) == 1:
				the_size = lens.pop()
				if self.force and the_size != self.batch_size:
					break
				elif the_size:
					yield self.wrap(result)

					batches_produced += 1
					if self.num_batches is not None:
						if batches_produced >= self.num_batches:
							break
				else:
					break
			else:
				if proceed:
					logger.error('Unequal length batches were produced: '
						'%s', self.wrap(result))
					raise ValueError('Managed to accumulate unequal-length '
						'batches, but we were still told to continue '
						'iterating. This is a bug.')

				smallest = min(lens)
				if smallest == 0:
					break
				if self.force and smallest != self.batch_size:
					break
				result = [x[:smallest] for x in result]
				yield self.wrap(result)

	###########################################################################
	def pre_iter(self):
		""" Pre-iteration hook.
		"""

		if not self.randomize or \
				self.shuffle_after or \
				not self.neighborhood_key:
			super().pre_iter()
			return

		logger.info('Calculating the nearest-neighbor shuffle indices using '
			'data source "%s".', self.neighborhood_key)

		# Create a local, complete copy of the sort-by data.
		# TODO: Can this safely be cached?
		data = numpy.empty(
			(len(self.neighborhood_data), ) + self.neighborhood_data.shape()
		)
		start = 0
		for batch in self.neighborhood_data:
			data[start:start+len(batch)] = batch[:]
			start += len(batch)

		# Determine the sort indices.
		indices = neighbor_sort.argsort(
			data,
			self.batch_size,
			self.neighborhood_size,
			self.neighborhood_growth
		)

		# Apply the indices.
		for source in self.sources:
			source.shuffle(indices)

		logger.debug('Finished shuffling.')

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

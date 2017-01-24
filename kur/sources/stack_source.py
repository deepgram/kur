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

import bisect

import numpy

from . import ChunkSource
from ..utils import partial_sum

###############################################################################
class StackSource(ChunkSource):
	""" A data source which is simple wrapper around an in-memory array.
	"""

	###########################################################################
	@classmethod
	def default_chunk_size(cls):
		""" Returns the default chunk size for this source.
		"""
		return ChunkSource.USE_BATCH_SIZE

	###########################################################################
	def __init__(self, *sources, **kwargs):
		""" Create a new stacked data source.

			# Arguments

			*sources: list of Source instances. The data sources to stack.
		"""
		super().__init__(**kwargs)

		if not sources:
			raise ValueError('Cannot stack zero sources.')

		self.sources = []
		for source in sources:
			self.stack(source)
		self.indices = None
		self.draw = None

	###########################################################################
	def __len__(self):
		""" Returns the number of samples this source provides.
		"""
		return sum(len(source) for source in self.sources)

	###########################################################################
	def stack(self, source):
		""" Adds a new source to this data stack.
		"""
		if source.is_derived():
			raise ValueError('Cannot stack derived sources.')
		if len(source) == 0:
			raise ValueError('Cannot stack sources with unknown lengths.')

		if self.sources:
			if len(source.shape()) != len(self.shape()):
				raise ValueError('Cannot stack sources with different shapes. '
					'We already had source(s) of shape {}, and now we were '
					'asked to stack a source of shape {}.'.format(
						self.shape(), source.shape()
					))
			for this_shape, that_shape in zip(self.shape(), source.shape()):
				if this_shape is None or that_shape is None:
					continue
				if this_shape != that_shape:
					raise ValueError('Incompatible shapes cannot be stacked. '
						'We already had source(s) of shape {}, and now we '
						'were asked to stack a source of shape {}.'.format(
							self.shape(), source.shape()
						))

		self.sources.append(source)
		self.indices = None

	###########################################################################
	def shape(self):
		""" Returns the shape of the individual data samples.
		"""
		return self.sources[0].shape()

	###########################################################################
	def can_shuffle(self):
		""" This source can be shuffled.
		"""
		return all(source.can_shuffle() for source in self.sources)

	###########################################################################
	def shuffle(self, indices):
		""" Applies a permutation to the data.
		"""
		if not self.can_shuffle():
			raise ValueError('This source is not shuffleable.')

		if len(indices) > len(self):
			raise ValueError('Shuffleable was asked to apply permutation, but '
				'the permutation is longer than the length of the data set.')

		if self.indices is None:
			self.indices = indices[:len(indices)]
		else:
			if len(self.indices) != len(indices):
				raise ValueError('The number of shuffle indices has changed '
					'unexpectedly. This can cause mis-aligned data and is not '
					'permitted.')
			self.indices = self.indices[indices]

		cutoffs = list(partial_sum(len(source) for source in self.sources))

		counters = [0 for _ in range(len(self.sources))]
		sub_shuffles = [[None]*len(source) for source in self.sources]
		draw = [None]*len(indices)
		for i, index in enumerate(indices):
			which = bisect.bisect(cutoffs, index)
			offset = cutoffs[which-1] if which else 0
			sub_shuffles[which][counters[which]] = index - offset
			counters[which] += 1
			draw[i] = which

		for i, x in enumerate(sub_shuffles):
			self.sources[i].shuffle(x)
		self.draw = draw

		# 1. Figure out the shufflings for each source.
		# 2. Shuffle the sources accordingly.
		# 3. Figure out the ordering we need to draw from each source.

	###########################################################################
	def __iter__(self):
		""" Returns the next chunk of data.
		"""

		queues = [None]*len(self.sources)
		iterators = [iter(source) for source in self.sources]

		if self.draw is None:
			self.draw = [i for i, source in enumerate(self.sources) \
				for _ in range(len(source))]

		if isinstance(self.chunk_size, int):
			chunk_size = self.chunk_size
		else:
			chunk_size = ChunkSource.DEFAULT_CHUNK_SIZE

		start = 0
		num_entries = len(self)
		while start < num_entries:
			if self.chunk_size is ChunkSource.ALL_ITEMS:
				end = num_entries
			else:
				end = min(num_entries, start + chunk_size)

			num = end - start
			result = numpy.empty(shape=(num,) + self.shape())
			for i in range(num):

				which = self.draw[start+i]

				if queues[which] is None:
					queues[which] = next(iterators[which])
				result[i] = queues[which][0]
				queues[which] = queues[which][1:]
				if len(queues[which]) == 0:
					queues[which] = None

			yield result
			start = end

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

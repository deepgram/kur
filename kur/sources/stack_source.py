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

import numpy

from . import ChunkSource

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

		self.draw = None
		self.accumulator = None
		self.sources = []
		for source in sources:
			self.stack(source)

	###########################################################################
	def __len__(self):
		""" Returns the number of samples this source provides.
		"""
		return sum(len(source) for source in self.sources)

	###########################################################################
	def __getattr__(self, name):
		""" Pass-through for stacked sources.
		"""
		for source in self.sources:
			if hasattr(source, name):
				return getattr(source, name)
		raise AttributeError('No such attribute: {}'.format(name))

	###########################################################################
	def stack(self, source):
		""" Adds a new source to this data stack.
		"""
		if source.is_derived():
			raise ValueError('Cannot stack derived sources.')
		if not source:
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

		new_block = numpy.ones(len(source), dtype=numpy.int32) \
			* len(self.sources)
		if self.draw is None:
			self.draw = new_block
		else:
			self.draw = numpy.concatenate((self.draw, new_block))

		new_block = numpy.arange(len(source))
		if self.accumulator is None:
			self.accumulator = new_block
		else:
			self.accumulator = numpy.concatenate((self.accumulator, new_block))

		self.sources.append(source)

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

		indices = indices[:len(self)]

		self.draw = self.draw[indices]
		ctr = self.accumulator[indices]

		sub_shuffles = [[None]*len(source) for source in self.sources]
		counters = [0 for _ in range(len(self.sources))]
		for i, draw in enumerate(self.draw):
			self.accumulator[i] = counters[draw]
			sub_shuffles[draw][counters[draw]] = ctr[i]
			counters[draw] += 1

		for i, x in enumerate(sub_shuffles):
			self.sources[i].shuffle(x)

	###########################################################################
	def __iter__(self):
		""" Returns the next chunk of data.
		"""

		for source in self.sources:
			if isinstance(source, ChunkSource):
				if source.requested_chunk_size is ChunkSource.USE_BATCH_SIZE:
					source.set_chunk_size(self.chunk_size)

		queues = [None]*len(self.sources)
		iterators = [iter(source) for source in self.sources]

		assert self.draw is not None

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
			result = [None] * num
			for i in range(num):

				which = self.draw[start+i]

				if queues[which] is None:
					queues[which] = next(iterators[which])
				result[i] = queues[which][0]
				queues[which] = queues[which][1:]
				if len(queues[which]) == 0:	 # pylint: disable=len-as-condition
					queues[which] = None

			yield numpy.array(result)
			start = end

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

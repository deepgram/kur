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

from . import ChunkSource

###############################################################################
class VanillaSource(ChunkSource):
	""" A data source which is simple wrapper around an in-memory array.
	"""

	###########################################################################
	@classmethod
	def default_chunk_size(cls):
		""" Returns the default chunk size for this source.
		"""
		return ChunkSource.ALL_ITEMS

	###########################################################################
	def __init__(self, data, *args, **kwargs):
		""" Create a new vanilla data provider.

			# Arguments

			data: numpy-like array. The data to provide.
			chunk_size: int (default: None). The maximum number of samples to
				return at each iteration, or None to return all samples.

			# Notes

			- This class will not "take ownership" of the dataset; that is, it
			  only holds a reference to the data. This means that you are free
			  to shuffle the underlying dataset (or even change the shape or
			  length of the dataset, although this isn't necessarily a good
			  idea).
		"""
		super().__init__(*args, **kwargs)

		self.data = data

	###########################################################################
	def __len__(self):
		""" Returns the number of samples this source provides.
		"""
		return self.data.shape[0]

	###########################################################################
	def shape(self):
		""" Returns the shape of the individual data samples.
		"""
		return self.data.shape[1:]

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
		self.data[:len(indices)] = self.data[indices]

	###########################################################################
	def __iter__(self):
		""" Returns the next chunk of data.
		"""

		start = 0
		num_entries = len(self)
		while start < num_entries:
			if self.chunk_size is ChunkSource.ALL_ITEMS:
				end = num_entries
			else:
				end = min(num_entries, start + self.chunk_size)

			yield self.data[start:end]
			start = end

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

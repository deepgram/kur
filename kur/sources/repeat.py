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

import numpy
from . import ChunkSource

###############################################################################
class RepeatSource(ChunkSource):
	""" A data source which always returns the same value.
	"""

	###########################################################################
	def __init__(self, value=None, *args, **kwargs):
		""" Create a new repeat data source.

			# Arguments

			value: object. The value to return for each sample.
			chunk_size: int (default: None). The maximum number of samples to
				return at each iteration, or None to choose some default
				number.
		"""
		super().__init__(*args, **kwargs)

		self.data = numpy.array([value for _ in range(self.chunk_size)])

	###########################################################################
	def __len__(self):
		""" Returns the number of samples this source provides.
		"""
		return 0

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
		pass

	###########################################################################
	def __iter__(self):
		""" Returns the next chunk of data.
		"""

		while True:
			yield self.data

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

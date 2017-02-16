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

import pytest
import numpy

from kur.sources import VanillaSource, StackSource, ChunkSource

###############################################################################
def flatten_source(source):
	""" Flattens a Source into a single numpy array.
	"""
	data = None
	start = 0
	for batch in source:
		if data is None:
			data = numpy.empty(
				(len(source), ) + source.shape(),
				dtype=batch.dtype
			)
		data[start:start+len(batch)] = batch[:]
		start += len(batch)

	return data

###############################################################################
@pytest.fixture(
	params=[20, 100, int(1.2 * ChunkSource.DEFAULT_CHUNK_SIZE)]
)
def num_entries(request):
	""" How many entries per Source
	"""
	return request.param

###############################################################################
@pytest.fixture(
	params=[None, 7]
)
def chunk_size(request):
	""" Set the chunk size for the source (or None to use the default).
	"""
	return request.param

###############################################################################
@pytest.fixture(
	params=[0, 1, 2]
)
def source(request, num_entries, chunk_size):
	""" Creates the data Source to test.
	"""
	num_stacks = request.param
	if num_stacks == 0:
		result = VanillaSource(numpy.random.permutation(num_entries)+10)
	else:
		result = StackSource(*[
			VanillaSource(
				numpy.random.permutation(num_entries) + 100*i
			) for i in range(num_stacks)
		])
	if chunk_size is not None:
		result.set_chunk_size(chunk_size)
	return result

###############################################################################
class TestShuffling:

	###########################################################################
	def test_shuffle(self, source):

		original_data = flatten_source(source)
		assert numpy.allclose(original_data, flatten_source(source))

		num_entries = len(source)
		current_permutation = numpy.arange(num_entries)

		for _ in range(5):
			indices = numpy.random.permutation(num_entries)
			source.shuffle(indices)

			current_permutation = current_permutation[indices]
			assert numpy.allclose(
				original_data[current_permutation],
				flatten_source(source)
			)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

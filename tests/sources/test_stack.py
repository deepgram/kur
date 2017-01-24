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

from kur.sources import VanillaSource, StackSource

###############################################################################
class TestStack:
	""" Tests for the StackSource.
	"""

	###########################################################################
	def test_single_fixed(self):
		""" Test a known permutation with exactly one source.
		"""
		data = numpy.array([1, 2, 3, 4, 5])
		indices = [4, 2, 1, 3, 0]
		expected = numpy.array([5, 3, 2, 4, 1])

		source = VanillaSource(data)
		stack = StackSource(source)
		stack.shuffle(indices)
		result = numpy.array([actual for batch in stack for actual in batch])

		assert numpy.allclose(result, expected)

	###########################################################################
	def test_single(self):
		""" Test a random permutation with exactly one source.
		"""
		data = numpy.array([1, 2, 3, 4, 5])
		source = VanillaSource(data)
		stack = StackSource(source)

		assert len(stack) == len(data)
		assert stack.shape() == ()

		indices = numpy.random.permutation(len(data))

		preshuffled = numpy.copy(data)
		stack.shuffle(indices)

		for i, x in enumerate(data):
			assert preshuffled[indices[i]] == x

		cur = 0
		for batch in stack:
			for actual in batch:
				expected = data[cur]
				assert actual == expected
				cur += 1

		assert cur == len(data)

	###########################################################################
	def test_double(self):
		""" Test a random permutation with exactly two sources.
		"""
		data = [
			numpy.array([1, 2, 3, 4, 5]),
			numpy.array([6, 7, 8, 9, 0])
		]
		sources = [
			VanillaSource(data[0]),
			VanillaSource(data[1])
		]
		stack = StackSource(*sources)

		total = sum(len(x) for x in data)
		assert len(stack) == total
		assert stack.shape() == ()

		indices = numpy.random.permutation(total)

		preshuffled = [numpy.copy(x) for x in data]
		stack.shuffle(indices)

		sub = [i for i in indices if i < len(data[0])]
		for i, x in enumerate(data[0]):
			assert preshuffled[0][sub[i]] == x

		sub = [i % len(data[0]) for i in indices if i >= len(data[0])]
		for i, x in enumerate(data[1]):
			assert preshuffled[1][sub[i]] == x

		cur = 0
		for batch in stack:
			for actual in batch:

				i = indices[cur]
				expected = preshuffled[i//5][i%5]
				assert actual == expected
				cur += 1

		assert cur == total

	###########################################################################
	def test_triple_fixed(self):
		""" Test a known permutation with exactly three sources.
		"""
		data = [
			numpy.arange(10, 16),
			numpy.arange(16, 20),
			numpy.arange(20, 23)
		]
		indices = [12, 3, 5, 6, 11, 0, 10, 2, 1, 4, 9, 7, 8]
		expected = numpy.array(
			[22, 13, 15, 16, 21, 10, 20, 12, 11, 14, 19, 17, 18]
		)

		sources = [VanillaSource(x) for x in data]
		stack = StackSource(*sources)
		stack.shuffle(indices)
		result = numpy.array([actual for batch in stack for actual in batch])

		assert numpy.allclose(result, expected)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

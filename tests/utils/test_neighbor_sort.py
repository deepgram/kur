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

import copy

import numpy
import pytest

from kur.utils import neighbor_sort

###############################################################################
@pytest.fixture
def simple_data():
	return numpy.array([201, 202, 303, 101, 203, 102, 302, 103, 301])

###############################################################################
@pytest.fixture(
	params=[True, False]
)
def lotsa_data(request):
	with_repeats = request.param
	return numpy.random.choice(1000, size=500, replace=with_repeats)

###############################################################################
@pytest.fixture(
	params=[True, False]
)
def use_uniform(request):
	return request.param

###############################################################################
class TestNeighborSort:

	###########################################################################
	def test_even(self, simple_data, use_uniform):
		before = copy.deepcopy(simple_data)
		indices = neighbor_sort.argsort(
			simple_data,
			batch_size=3,
			neighborhood=5,
			uniform=use_uniform
		)

		# Make sure nothing changed.
		assert numpy.allclose(before, simple_data)

		# Do we have a valid permutation?
		assert indices.shape[0] == len(simple_data)
		assert indices.ndim == 1
		assert numpy.allclose(sorted(indices), numpy.arange(len(simple_data)))

		# Test the permutation.
		after = simple_data[indices]

		# Partition the result into three groups of three items each.
		groups = [after[3*i:(i+1)*3] for i in range(3)]
		# First sort by 100s.
		groups.sort(key=lambda x: x[0])
		# Now sort the items in each group.
		groups = [sorted(x) for x in groups]
		# Flatten the array.
		result = numpy.ravel(groups)

		# This should be fully sorted.
		assert numpy.allclose(result, sorted(before))

	###########################################################################
	def test_uniqueness(self, lotsa_data, use_uniform):
		indices = neighbor_sort.argsort(
			lotsa_data,
			batch_size=13,
			neighborhood=5,
			uniform=use_uniform
		)

		assert len(indices) == len(lotsa_data)
		assert indices.ndim == 1
		assert numpy.allclose(sorted(indices), numpy.arange(len(lotsa_data)))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

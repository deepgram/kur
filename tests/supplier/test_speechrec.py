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

from unittest.mock import Mock
from functools import partial

import pytest

import numpy

from kur.supplier.speechrec import SpeechRecognitionSupplier

@pytest.fixture
def seed():
	return 0

@pytest.fixture
def num_entries():
	return 50

@pytest.fixture
def fake_data(num_entries):
	return list(range(num_entries))

@pytest.fixture
def fake_supplier(seed, fake_data):
	result = Mock()
	result.metadata = {'entries' : len(fake_data)}
	result.data = {'data' : fake_data}
	result.kurfile.get_seed.return_value = seed
	result.downselect = partial(SpeechRecognitionSupplier.downselect, result)
	return result

@pytest.fixture
def permuted_numbers(seed, num_entries):
	return numpy.random.RandomState(seed).permutation(num_entries)

def select_from(data, indices):
	if isinstance(data, numpy.ndarray):
		return data[sorted(indices)]

	indices = set(indices)
	return type(data)(x for i, x in enumerate(data) if i in indices)

class TestSpeechRecognitionSupplier:

	def verify(self, supplier, expected_length, expected_data):
		assert supplier.metadata['entries'] == expected_length
		assert numpy.allclose(supplier.data['data'], expected_data)

	def test_sample_none(self, fake_supplier, fake_data):
		fake_supplier.downselect(None)
		self.verify(fake_supplier, len(fake_data), fake_data)

	def test_sample_int(self, fake_supplier, fake_data, permuted_numbers):
		fake_supplier.downselect(10)
		self.verify(fake_supplier, 10,
			select_from(fake_data, permuted_numbers[:10]))

	def test_sample_percent(self, fake_supplier, fake_data, permuted_numbers):
		fake_supplier.downselect('10%')
		self.verify(fake_supplier, 5,
			select_from(fake_data, permuted_numbers[:5]))

	def test_sample_no_end(self, fake_supplier, fake_data, permuted_numbers):
		fake_supplier.downselect('10-')
		self.verify(fake_supplier, 40,
			select_from(fake_data, permuted_numbers[10:]))

	def test_sample_range(self, fake_supplier, fake_data, permuted_numbers):
		fake_supplier.downselect('10-20')
		self.verify(fake_supplier, 10,
			select_from(fake_data, permuted_numbers[10:20]))

	def test_sample_range_percent(self, fake_supplier, fake_data,
			permuted_numbers):
		fake_supplier.downselect('10-20%')
		self.verify(fake_supplier, 5,
			select_from(fake_data, permuted_numbers[5:10]))

	def test_sample_range_percent_no_end(self, fake_supplier, fake_data,
			permuted_numbers):
		fake_supplier.downselect('10-%')
		self.verify(fake_supplier, 45,
			select_from(fake_data, permuted_numbers[5:]))

	def test_sample_lots(self, fake_supplier, fake_data,
			permuted_numbers):
		fake_supplier.downselect('10000')
		self.verify(fake_supplier, 50,
			select_from(fake_data, permuted_numbers))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

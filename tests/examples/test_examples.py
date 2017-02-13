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

import os

import pytest

from kur.kurfile import Kurfile

from modkurfile import modify_kurfile
from example import example

@pytest.fixture
def example_directory():
	return os.path.normpath(
		os.path.join(
			os.path.dirname(__file__),
			'../..',
			'examples'
		)
	)

@pytest.fixture(
	params=['mnist.yml', 'cifar.yml', 'speech.yml']
)
def kurfile(request, example_directory, jinja_engine):
	result = Kurfile(
		os.path.join(example_directory, request.param),
		jinja_engine
	)

	modify_kurfile(result.data)
	for k in ('train', 'validate', 'test', 'evaluate'):
		if k in result.data and 'data' in result.data[k]:
			for data_source in result.data[k]['data']:
				if 'speech_recognition' in data_source \
					and 'normalization' in data_source['speech_recognition']:
					del data_source['speech_recognition']['normalization']

	result.parse()
	return result

class TestExample:

	@example
	def test_train(self, kurfile):
		if 'train' not in kurfile.data:
			pytest.skip('No training section defined for this Kurfile.')
		func = kurfile.get_training_function()
		func()

	@example
	def test_test(self, kurfile):
		if 'test' not in kurfile.data:
			pytest.skip('No testing section defined for this Kurfile.')
		func = kurfile.get_testing_function()
		func()

	@example
	def test_evaluate(self, kurfile):
		if 'evaluate' not in kurfile.data:
			pytest.skip('No evaluation section defined for this Kurfile.')
		func = kurfile.get_evaluation_function()
		func()

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

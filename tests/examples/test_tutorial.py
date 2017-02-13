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

import textwrap
import pickle
import os
import tempfile

import yaml
import pytest
import numpy

from kur.kurfile import Kurfile

from modkurfile import modify_kurfile
from example import example

@pytest.fixture(scope='session')
def tutorial_yaml():
	return textwrap.dedent("""
		model:
		  - input: point
		  - dense: 128
		  - activation: tanh
		  - dense: 1
		  - activation: tanh
		    name: above

		train:
		  data:
		    - pickle: train.pkl
		  epochs: 10
		  weights: best.w
		  log: tutorial-log

		validate:
		  data:
		    - pickle: validate.pkl
		  weights: best.w

		test:
		  data:
		    - pickle: test.pkl
		  weights: best.w

		evaluate:
		  data:
		    - pickle: evaluate.pkl
		  weights: best.w
		  destination: output.pkl

		loss:
		  - target: above
		    name: mean_squared_error
	""")

@pytest.fixture(scope='session')
def num_samples():
	return 10

def make_points(num_samples, filename):
	x = numpy.array([
		numpy.random.uniform(-numpy.pi, numpy.pi, num_samples),
		numpy.random.uniform(-1, 1, num_samples)
	]).T
	y = (numpy.sin(x[:,0]) < x[:,1]).astype(numpy.float32) * 2 - 1
	with open(filename, 'wb') as fh:
		fh.write(pickle.dumps({'point' : x, 'above' : y}))

@pytest.fixture(scope='session')
def kurfile(tutorial_yaml):
	result = Kurfile(yaml.load(tutorial_yaml))

	modify_kurfile(result.data)
	for k in ('train', 'validate', 'test', 'evaluate'):
		if k in result.data and 'data' in result.data[k]:
			result.data[k]['data'][0]['pickle'] = os.path.join(
				tempfile.gettempdir(),
				result.data[k]['data'][0]['pickle']
			)

	result.parse()
	return result

@pytest.fixture(scope='session')
def tutorial_data(request, kurfile, num_samples):
	files = []
	for k in ('train', 'validate', 'test', 'evaluate'):
		if k not in kurfile.data:
			continue
		if 'data' not in kurfile.data[k]:
			continue
		if not kurfile.data[k]['data']:
			continue
		files.append(kurfile.data[k]['data'][0]['pickle'])

	for filename in files:
		make_points(num_samples, filename)

	yield

	for filename in files:
		os.unlink(filename)

class TestTutorial:

	@example
	def test_train(self, kurfile, tutorial_data):
		kurfile.get_training_function()()

	@example
	def test_test(self, kurfile, tutorial_data):
		kurfile.get_testing_function()()

	@example
	def test_evaluate(self, kurfile, tutorial_data):
		kurfile.get_evaluation_function()()

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

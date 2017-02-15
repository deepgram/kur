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

import pytest
import numpy

from kur.utils import get_subclasses

from kur.engine import Engine, JinjaEngine
from kur.backend import Backend
from kur.containers import Container
from kur.model import Model
from kur.providers import BatchProvider
from kur.sources import VanillaSource
from kur.utils import can_import

###############################################################################
def pytest_addoption(parser):
    parser.addoption("--examples", action="store_true",
		help="Test the examples and tutorials as well.")

################################################################################
def keras_mock(cls, backend, deps, **kwargs):
	""" Fakes a backend-looking object that can be used in place of a Keras
		backend. It is used to force certain variations of the Keras backend to
		get tested.
	"""
	result = lambda: cls(backend=backend, **kwargs)
	setattr(result, 'is_supported',
		lambda: cls.is_supported() and all(can_import(dep) for dep in deps))
	setattr(result, 'get_name',
		lambda: 'mock_{}_{}'.format(cls.get_name(), backend))
	return result

###############################################################################
def enumerate_backends():
	""" Enumerates all the backends.

		Instead of calling `Backend.get_all_backends()` directly, this layer of
		indirection allows us to "fake" particular backend variants.
	"""
	result = []

	all_backends = Backend.get_all_backends(supported_only=False)
	for backend in all_backends:
		if backend is Backend.get_backend_by_name('keras'):
			result.extend([
				keras_mock(backend, 'theano', ('theano', ), optimizer=False),
				keras_mock(backend, 'tensorflow', ('tensorflow', ))
			])
		else:
			result.append(backend)

	return result

###############################################################################
@pytest.fixture(
	params=enumerate_backends()
)
def a_backend(request):
	""" Fixture for obtaining a backend.
	"""
	cls = request.param
	if not cls.is_supported():
		pytest.xfail('Backend {} is not installed and cannot be tested.'
			.format(cls.get_name()))
	return cls()

###############################################################################
@pytest.fixture(
	params=get_subclasses(Engine, recursive=True)
)
def an_engine(request):
	""" Fixture for obtaining an engine.
	"""
	return request.param()

###############################################################################
@pytest.fixture
def passthrough_engine():
	""" Returns a Jinja2 engine.
	"""
	return JinjaEngine()

###############################################################################
@pytest.fixture
def jinja_engine():
	""" Returns a Jinja2 engine.
	"""
	return JinjaEngine()

###############################################################################
def model_with_containers(backend, containers):
	""" Convenience function for creating a model instance from the toy model
		functions in this module (e.g., `simple_model()`).
	"""
	return Model(
		backend=backend,
		containers=[
			Container.create_container_from_data(container)
			for container in containers
		]
	)

###############################################################################
@pytest.fixture
def simple_model(a_backend):
	""" Returns a model that is extremely simple (one layer).
	"""
	return model_with_containers(
		backend=a_backend,
		containers=[
			{'input' : {'shape' : [10]}, 'name' : 'TEST_input'},
			{'dense' : 1, 'name' : 'TEST_output'},
		]
	)

###############################################################################
@pytest.fixture
def simple_data():
	""" Returns a small provider that can be used to train the `simple_model()`.
	"""
	return BatchProvider(
		sources={
			'TEST_input' : VanillaSource(numpy.random.uniform(size=(100, 10))),
			'TEST_output' : VanillaSource(numpy.random.uniform(size=(100, 1)))
		}
	)

###############################################################################
@pytest.fixture
def ctc_model(a_backend):
	""" Returns a model which uses the CTC loss function.
	"""
	output_timesteps = 10
	vocab_size = 4
	return model_with_containers(
		backend=a_backend,
		containers=[
			{'input' : {'shape' : [output_timesteps, 2]}, 'name' : 'TEST_input'},
			{'recurrent' : {'size' : vocab_size+1, 'sequence' : True}},
			{'activation' : 'softmax', 'name' : 'TEST_output'}
		]
	)

###############################################################################
@pytest.fixture
def ctc_data():
	""" Returns a provider that can be used with `ctc_model()`.
	"""
	number_of_samples = 11
	vocab_size = 4	# Same as above
	output_timesteps = 10
	maximum_transcription_length = 4	# Must be <= output_timesteps
	return BatchProvider(
		sources={
			'TEST_input' : VanillaSource(numpy.random.uniform(
				low=-1, high=1, size=(number_of_samples, output_timesteps, 2)
			)),
			'TEST_transcription' : VanillaSource(numpy.random.random_integers(
				0, vocab_size-1,
				size=(number_of_samples, maximum_transcription_length)
			)),
			'TEST_input_length' : VanillaSource(numpy.ones(
				shape=(number_of_samples, 1)
			) * output_timesteps),
			'TEST_transcription_length' : VanillaSource(
				numpy.random.random_integers(1, maximum_transcription_length,
				size=(number_of_samples, 1)
			))
		}
	)

###############################################################################
@pytest.fixture
def uber_model(a_backend):
	""" One model to rule them all...
		Every single container type should be represented here.
	"""
	return model_with_containers(
		backend=a_backend,
		containers=[
			# Force it to infer the shape from data.
			{'input' : 'TEST_input'},
			# Shape: (32, 32)
			{'transpose' : [0, 1]},
			{'expand' : -1},
			{'debug' : 'hello world'},
			# Shape: (32, 32, 1)
			{'for' : {'with_index' : 'idx', 'range' : 2, 'iterate' : [
				{'convolution' : {'kernels' : '{{ 2*(idx+1) }}', 'size' : [2, 2]}},
				# Shape: (32, 32, 2), (16, 16, 4)
				{'activation' : 'relu'},
				{'pool' : {'size' : [2, 2], 'strides' : 2}},
				# Shape: (16, 16, 2), (8, 8, 4)
				{'assert' : '{{ idx < 2 }}'}
			]}},
			{'dropout' : 0.2},
			{'parallel' : {'apply' : [
				'flatten',
				# Shape: (8, 32)
				{'dense' : {'size' : 10}, 'name' : 'TEST_reuse'}
				# Shape: (8, 10)
			]}},
			{'recurrent' : {'size' : 32}},
			# Shape: (8, 32)
			'batch_normalization',
			{'reuse' : 'TEST_reuse'},
			# Shape: (8, 10)
			{'flatten' : None, 'name' : 'TEST_mark1'},
			{'dense' : 80, 'name' : 'TEST_mark2'},
			{'merge' : 'average', 'inputs' : ['TEST_mark1', 'TEST_mark2']},
			# Shape: (80, )
			{'output' : 'TEST_output'}
		]
	)

###############################################################################
@pytest.fixture
def uber_data():
	""" In the land of Mordor, where the shadows lie.
		Data for the uber model.
	"""
	return BatchProvider(
		sources={
			'TEST_input' : VanillaSource(numpy.random.uniform(
				low=-1, high=1, size=(2, 32, 32)
			)),
			'TEST_output' : VanillaSource(numpy.random.uniform(
				low=-1, high=1, size=(2, 80)
			))
		}
	)

###############################################################################
@pytest.fixture(
	params=[
		(simple_model, simple_data),
		(ctc_model, ctc_data)
	]
)
def any_model_with_data(request, a_backend):
	""" Returns a few different models with their data.
	"""
	model, data = request.param
	return (model(a_backend), data())

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

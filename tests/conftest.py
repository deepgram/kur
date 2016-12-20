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

from kur.engine import Engine
from kur.backend import Backend
from kur.containers import Container
from kur.model import Model
from kur.providers import BatchProvider
from kur.sources import VanillaSource
from kur.utils import can_import

################################################################################
def keras_mock(cls, backend, deps):
	""" Fakes a backend-looking object that can be used in place of a Keras
		backend. It is used to force certain variations of the Keras backend to
		get tested.
	"""
	result = lambda: cls(backend=backend)
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
				keras_mock(backend, 'theano', ('theano', )),
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

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

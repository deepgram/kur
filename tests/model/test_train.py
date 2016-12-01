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

from kur.loss import MeanSquaredError
from kur.optimizer import Adam
from kur.model import Trainer

@pytest.fixture
def loss():
	return MeanSquaredError()

@pytest.fixture
def optimizer():
	return Adam()

################################################################################
class TestTrain:
	""" Tests for the Trainer
	"""

	############################################################################
	def test_trainer_with_optimizer(self, simple_model, loss, optimizer):
		""" Tests if we can compile a Trainer instance with an optimizer.
		"""
		simple_model.parse(None)
		simple_model.build()
		trainer = Trainer(
			model=simple_model,
			loss=loss,
			optimizer=optimizer
		)
		trainer.compile()

	############################################################################
	def test_trainer_without_optimizer(self, simple_model, loss, optimizer):
		""" Tests if we can compile a Trainer instance without an optimizer.
		"""
		simple_model.parse(None)
		simple_model.build()
		trainer = Trainer(
			model=simple_model,
			loss=loss,
			optimizer=optimizer
		)
		trainer.compile()

	############################################################################
	def test_training(self, simple_model, loss, optimizer, simple_data):
		""" Tests that a single training epoch can succeed.
		"""
		simple_model.parse(None)
		simple_model.build()
		trainer = Trainer(
			model=simple_model,
			loss=loss,
			optimizer=optimizer
		)
		trainer.train(provider=simple_data, epochs=1)

	############################################################################
	def test_testing(self, simple_model, loss, simple_data):
		""" Tests that a testing run can succeed.
		"""
		simple_model.parse(None)
		simple_model.build()
		trainer = Trainer(
			model=simple_model,
			loss=loss
		)
		trainer.test(provider=simple_data)

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

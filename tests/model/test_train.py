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

from kur.loss import MeanSquaredError, Ctc
from kur.optimizer import Adam
from kur.model import Trainer

@pytest.fixture
def loss():
	return MeanSquaredError()

@pytest.fixture
def ctc_loss():
	return Ctc(
		input_length='TEST_input_length',
		output_length='TEST_transcription_length',
		output='TEST_transcription'
	)

@pytest.fixture
def optimizer():
	return Adam()

###############################################################################
class TestTrain:
	""" Tests for the Trainer
	"""

	###########################################################################
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

	###########################################################################
	def test_trainer_without_optimizer(self, simple_model, loss):
		""" Tests if we can compile a Trainer instance without an optimizer.
		"""
		simple_model.parse(None)
		simple_model.build()
		trainer = Trainer(
			model=simple_model,
			loss=loss
		)
		trainer.compile()

	###########################################################################
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

	###########################################################################
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

	###########################################################################
	def test_ctc_train(self, ctc_model, ctc_data, ctc_loss, optimizer):
		""" Tests that we can compile and train a model using the CTC loss
			function.
		"""
		ctc_model.parse(None)
		ctc_model.register_provider(ctc_data)
		ctc_model.build()
		trainer = Trainer(
			model=ctc_model,
			loss=ctc_loss,
			optimizer=optimizer
		)
		trainer.train(provider=ctc_data, epochs=1)

	###########################################################################
	def test_ctc_test(self, ctc_model, ctc_data, ctc_loss):
		""" Tests that we can compile and test a model using the CTC loss
			function.
		"""
		ctc_model.parse(None)
		ctc_model.register_provider(ctc_data)
		ctc_model.build()
		trainer = Trainer(
			model=ctc_model,
			loss=ctc_loss
		)
		trainer.test(provider=ctc_data)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

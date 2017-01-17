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
from kur.model import Executor
from kur.providers import BatchProvider

###############################################################################
@pytest.fixture
def ctc_eval_data(ctc_data):
	""" Provides CTC data with only the evaluation-time input data available.
	"""
	for key, source in zip(ctc_data.keys, ctc_data.sources):
		if key == 'TEST_input':
			return BatchProvider(sources={key : source})

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
class TestExecutor:
	""" Tests for the Executor (training/validation/testing).
	"""

	###########################################################################
	def test_trainer_with_optimizer(self, simple_model, loss, optimizer):
		""" Tests if we can compile a Executor instance with an optimizer.
		"""
		simple_model.parse(None)
		simple_model.build()
		trainer = Executor(
			model=simple_model,
			loss=loss,
			optimizer=optimizer
		)
		trainer.compile()

	###########################################################################
	def test_trainer_without_optimizer(self, simple_model, loss):
		""" Tests if we can compile a Executor instance without an optimizer.
		"""
		simple_model.parse(None)
		simple_model.build()
		trainer = Executor(
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
		trainer = Executor(
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
		trainer = Executor(
			model=simple_model,
			loss=loss
		)
		trainer.test(provider=simple_data)

	###########################################################################
	def test_evaluator(self, simple_model):
		""" Tests if we can compile an Executor instance.
		"""
		simple_model.parse(None)
		simple_model.build()
		evaluator = Executor(
			model=simple_model
		)
		evaluator.compile()

	###########################################################################
	def test_evaluating(self, simple_model, simple_data):
		""" Tests that an evaluation run can succeed.
		"""
		simple_model.parse(None)
		simple_model.build()
		evaluator = Executor(
			model=simple_model
		)
		evaluator.evaluate(provider=simple_data)

	###########################################################################
	def test_ctc_train(self, ctc_model, ctc_data, ctc_loss, optimizer):
		""" Tests that we can compile and train a model using the CTC loss
			function.
		"""
		ctc_model.parse(None)
		ctc_model.register_provider(ctc_data)
		ctc_model.build()
		trainer = Executor(
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
		trainer = Executor(
			model=ctc_model,
			loss=ctc_loss
		)
		trainer.test(provider=ctc_data)

	###########################################################################
	def test_ctc_evaluating(self, ctc_model, ctc_eval_data):
		""" Tests that we can evaluate a model that was trained with CTC loss.
		"""
		ctc_model.parse(None)
		ctc_model.register_provider(ctc_eval_data)
		ctc_model.build()
		evaluator = Executor(
			model=ctc_model
		)
		evaluator.evaluate(provider=ctc_eval_data)

	###########################################################################
	def test_uber_train(self, uber_model, uber_data, jinja_engine, loss,
		optimizer):
		""" Tests that we can compile and train a diverse model.
		"""
		uber_model.parse(jinja_engine)
		uber_model.register_provider(uber_data)
		uber_model.build()
		trainer = Executor(
			model=uber_model,
			loss=loss,
			optimizer=optimizer
		)
		trainer.compile()
		trainer.train(provider=uber_data, epochs=1)

	###########################################################################
	def test_uber_test(self, uber_model, uber_data, jinja_engine, loss):
		""" Tests that we can compile and test a diverse model.
		"""
		uber_model.parse(jinja_engine)
		uber_model.register_provider(uber_data)
		uber_model.build()
		trainer = Executor(
			model=uber_model,
			loss=loss
		)
		trainer.compile()
		trainer.test(provider=uber_data)

	###########################################################################
	def test_uber_evaluating(self, uber_model, uber_data, jinja_engine):
		""" Tests that we can evaluate a very diverse model.
		"""
		uber_model.parse(jinja_engine)
		uber_model.register_provider(uber_data)
		uber_model.build()
		evaluator = Executor(
			model=uber_model
		)
		evaluator.evaluate(provider=uber_data)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

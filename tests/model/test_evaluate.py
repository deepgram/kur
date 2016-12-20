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

from kur.model import Evaluator
from kur.providers import BatchProvider

###############################################################################
@pytest.fixture
def ctc_eval_data(ctc_data):
	""" Provides CTC data with only the evaluation-time input data available.
	"""
	for key, source in zip(ctc_data.keys, ctc_data.sources):
		if key == 'TEST_input':
			return BatchProvider(sources={key : source})

###############################################################################
class TestEvaluate:
	""" Tests for the Evaluator
	"""

	###########################################################################
	def test_evaluator(self, simple_model):
		""" Tests if we can compile an Evaluator instance.
		"""
		simple_model.parse(None)
		simple_model.build()
		evaluator = Evaluator(
			model=simple_model
		)
		evaluator.compile()

	###########################################################################
	def test_evaluating(self, simple_model, simple_data):
		""" Tests that an evaluation run can succeed.
		"""
		simple_model.parse(None)
		simple_model.build()
		evaluator = Evaluator(
			model=simple_model
		)
		evaluator.evaluate(provider=simple_data)

	###########################################################################
	def test_ctc_evaluating(self, ctc_model, ctc_eval_data):
		""" Tests that we can evaluate a model that was trained with CTC loss.
		"""
		ctc_model.parse(None)
		ctc_model.register_provider(ctc_eval_data)
		ctc_model.build()
		evaluator = Evaluator(
			model=ctc_model
		)
		evaluator.evaluate(provider=ctc_eval_data)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

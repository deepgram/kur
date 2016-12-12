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

from kur.model import Evaluator

###############################################################################
class TestEvaluate:
	""" Tests for the Evaluator
	"""

	###########################################################################
	def test_evaluator(self, simple_model):
		""" Tests if we can compile a Trainer instance without an optimizer.
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

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

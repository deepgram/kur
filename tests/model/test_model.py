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

###############################################################################
class TestModel:
	""" Tests for the Model
	"""

	###########################################################################
	def test_all(self, any_model_with_data, an_engine):
		""" Tests if we can assemble a handful of different simple models.
		"""
		a_model, _ = any_model_with_data
		a_model.parse(an_engine)
		a_model.build()

	###########################################################################
	def test_uber(self, uber_model, uber_data, jinja_engine):
		""" Tests if we can assemble the uber model.
		"""
		uber_model.parse(jinja_engine)
		uber_model.register_provider(uber_data)
		uber_model.build()

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

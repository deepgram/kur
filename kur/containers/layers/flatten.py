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

from . import Layer

################################################################################
class Flatten(Layer):					# pylint: disable=too-few-public-methods
	""" A layer which flattens the input tensor.
	"""

	############################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new flattening layer.
		"""
		super().__init__(*args, **kwargs)

	############################################################################
	def _build(self, backend):
		""" Create the backend-specific placeholder.
		"""
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error
			yield L.Flatten(
				name=self.name
			)

		else:
			raise ValueError(
				'Unknown or unsupported backend: {}'.format(backend))

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

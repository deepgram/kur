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

###############################################################################
class Flatten(Layer):					# pylint: disable=too-few-public-methods
	""" A layer which flattens the input tensor.
	"""

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new flattening layer.
		"""
		super().__init__(*args, **kwargs)

	###########################################################################
	def _build(self, model):
		""" Create the backend-specific placeholder.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error
			yield L.Flatten(
				name=self.name
			)

		elif backend.get_name() == 'pytorch':

			from kur.backend.pytorch.modules import flatten

			def connect(inputs):
				""" Connects the layer.
				"""
				assert len(inputs) == 1
				return {
					'shape' : self.shape([inputs[0]['shape']]),
					'layer' : model.data.add_operation(flatten)(
						inputs[0]['layer']
					)
				}

			yield connect

		else:
			raise ValueError(
				'Unknown or unsupported backend: {}'.format(backend))

	###########################################################################
	def shape(self, input_shapes):
		""" Returns the output shape of this layer for a given input shape.
		"""
		if len(input_shapes) > 1:
			raise ValueError('Flatten layers only take a single input.')
		input_shape = input_shapes[0]

		result = 1
		for x in input_shape:
			result *= x
		return (result, )

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

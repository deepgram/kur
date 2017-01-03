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

from . import Layer, ParsingError

###############################################################################
class Activation(Layer):				# pylint: disable=too-few-public-methods
	""" An activation layer.

		Some layers may include an 'activation' keyword. Those layers are
		intended to be equivalent to no-activation followed by this explicit
		activation layer.
	"""

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new activation layer.
		"""
		super().__init__(*args, **kwargs)
		self.type = None

	###########################################################################
	def _parse(self, engine):
		""" Parse the layer.
		"""

		super()._parse(engine)
		self.type = self.args

	###########################################################################
	def _build(self, model):
		""" Create the backend-specific placeholder.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error
			yield L.Activation(
				'linear' if self.type == 'none' or self.type is None \
					else self.type,
				name=self.name
			)

		else:
			raise ValueError(
				'Unknown or unsupported backend: {}'.format(backend))

	###########################################################################
	def shape(self, input_shapes):
		""" Returns the output shape of this layer for a given input shape.
		"""
		if len(input_shapes) > 1:
			raise ValueError('Activations only take a single input.')
		input_shape = input_shapes[0]
		return input_shape

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

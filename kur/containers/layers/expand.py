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
class Expand(Layer):					# pylint: disable=too-few-public-methods
	""" An expand layer adds a dimension of size 1.

		# Usage

		```
		expand:
		  dimension: 0
		```

		`DIM` must be an integer. It can be negative to count from the end.
	"""

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new expand layer.
		"""
		super().__init__(*args, **kwargs)
		self.dimension = None

	###########################################################################
	def _parse(self, engine):
		""" Parses out the expand layer.
		"""

		# Always call the parent.
		super()._parse(engine)

		if isinstance(self.args, dict) and 'dimension' in self.args:
			self.dimension = engine.evaluate(self.args['dimension'])
		elif isinstance(self.args, (str, int)):
			self.dimension = self.args
		else:
			raise ParsingError('The arguments to the "expand" layer must be an '
				'integer, or dictionary with a "dimension" key and an integer '
				'value.')
		try:
			self.dimension = int(self.dimension)
		except ValueError:
			raise ParsingError('"dimension" must evaluate to an integer.')

	###########################################################################
	def _build(self, model):
		""" Instantiates the layer with the given backend.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error
			import keras.backend as K			# pylint: disable=import-error

			target_dim = self.dimension
			if target_dim >= 0:
				target_dim += 1

			def expand_shape(input_shape):
				""" Computes the expanded shape.
				"""
				dim = target_dim
				if dim < 0:
					dim += len(input_shape) + 1
				return input_shape[:dim] + (1,) + input_shape[dim:]

			yield L.Lambda(
				lambda x: K.expand_dims(x, dim=target_dim),
				expand_shape,
				name=self.name
			)

		else:
			raise ValueError('Unknown or unsupported backend: {}'.format(backend))

	###########################################################################
	def shape(self, input_shapes):
		""" Returns the output shape of this layer for a given input shape.
		"""
		if len(input_shapes) > 1:
			raise ValueError('Activations only take a single input.')
		input_shape = input_shapes[0]

		dim = self.dimension
		if dim < 0:
			dim += len(input_shape) + 1
		if dim > len(input_shape) or dim < 0:
			raise ValueError('Invalid input shape for expand layer with '
				'dimension {}: {}'.format(self.dimension, input_shape))
		return input_shape[:dim] + (1, ) + input_shape[dim:]

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

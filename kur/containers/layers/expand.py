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

################################################################################
class Expand(Layer):					# pylint: disable=too-few-public-methods
	""" An expand layer adds a dimension of size 1.

		# Usage

		```
		expand:
		  dimension: 0
		```

		`DIM` must be an integer. It can be negative to count from the end.
	"""

	############################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new expand layer.
		"""
		super().__init__(*args, **kwargs)
		self.dimension = None

	############################################################################
	def _parse(self, engine):
		""" Parses out the expand layer.
		"""

		# Always call the parent.
		super()._parse(engine)

		# Parse self
		if 'dimension' not in self.args:
			raise ParsingError('Missing "dimension" key from expand container.')
		self.dimension = engine.evaluate(self.args['dimension'])
		try:
			self.dimension = int(self.dimension)
		except ValueError:
			raise ParsingError('"dimension" must evaluate to an integer.')

	############################################################################
	def _build(self, backend):
		""" Instantiates the layer with the given backend.
		"""
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

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

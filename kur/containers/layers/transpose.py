"""
Copyright 2017 Deepgram

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
class Transpose(Layer):				# pylint: disable=too-few-public-methods
	""" A transpose layer changes the ordering of the dimensions.

		# Usage

		```
		transpose: [0, 1]
		```

		or

		```
		transpose:
		  axes: [0, 1, 2]
		  include_batch: yes
		```
	"""

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new transpose layer.
		"""
		super().__init__(*args, **kwargs)
		self.axes = None
		self.include_batch = False

	###########################################################################
	def _parse(self, engine):
		""" Parses out the transpose layer.
		"""

		# Always call the parent.
		super()._parse(engine)

		if isinstance(self.args, (list, tuple)):
			self.axes = self.args
			self.include_batch = False
		elif isinstance(self.args, dict):
			if 'axes' in self.args:
				self.axes = engine.evaluate(self.args['axes'])
			else:
				raise ParsingError('Missing required key "axes" for transpose '
					'layer.')
			if 'include_batch' in self.args:
				self.include_batch = engine.evaluate(
					self.args['include_batch'])
			else:
				self.include_batch = False

		if not isinstance(self.axes, (list, tuple)):
			raise ParsingError('Value for "axes" in transpose layer must be a '
				'list or tuple. Received: {}'.format(self.axes))

		if not isinstance(self.include_batch, bool):
			raise ParsingError('Value for "batch" in transpose layer must be a '
				'boolean. Received: {}'.format(self.include_batch))

		for axis in self.axes:
			if not isinstance(axis, int):
				raise ParsingError('All axes in transpose layer must be '
					'integers. Received: {}'.format(axis))
			if axis < 0:
				raise ParsingError('All axes in transpose layer must be >= 0.')
		if set(self.axes) != set(range(len(self.axes))):
			raise ParsingError('Invalid axes transposition: {}'
				.format(self.axes))

	###########################################################################
	def _build(self, model):
		""" Instantiates the layer with the given backend.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error
			import keras.backend as K			# pylint: disable=import-error

			if self.include_batch:
				dims = self.axes
			else:
				dims = (0, ) + tuple(x+1 for x in self.axes)

			def transpose_shape(input_shape):
				""" Computes the expanded shape.
				"""
				return tuple(input_shape[i] for i in dims)

			yield L.Lambda(
				lambda x: K.permute_dimensions(x, dims),
				transpose_shape,
				name=self.name
			)

		else:
			raise ValueError('Unknown or unsupported backend: {}'
				.format(backend))

	###########################################################################
	def shape(self, input_shapes):
		""" Returns the output shape of this layer for a given input shape.
		"""
		if len(input_shapes) > 1:
			raise ValueError('Transpose layers only take a single input.')
		input_shape = input_shapes[0]

		if self.include_batch:
			return tuple(input_shape[i-1] for i in self.axes if i > 0)
		return tuple(input_shape[i] for i in self.axes)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

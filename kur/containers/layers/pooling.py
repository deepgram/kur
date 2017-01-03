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
class Pooling(Layer):				# pylint: disable=too-few-public-methods
	""" A pooling layer

		# Properties

		size: int or list of ints (required). The size of each kernel. If an
			integer is used, it is interpretted as a one-dimensional
			convolution, the same as if it were put into a length-1 list.
		strides: int or list of ints (optional; default: 1 in each dimension).
			The stride (subsampling rate) between convolutions. If a list, it
			must be the same length as `size` and specify the stride in each
			respective dimension. If a single number, it is used as the stride
			in each dimension.
		pool: one of (max, average). The pool function to apply.

		# Example

		```
		pool:
		  size: [2, 2]
		  strides: 1
		  type: max
		```
	"""

	POOL_TYPES = ('max', 'average')

	###########################################################################
	@classmethod
	def get_container_name(cls):
		""" Returns the name of the container class.
		"""
		return 'pool'

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new pooling layer.
		"""
		super().__init__(*args, **kwargs)
		self.size = None
		self.strides = None
		self.pooltype = None

	###########################################################################
	def _parse(self, engine):
		""" Parses out the pooling layer.
		"""
		# Always call the parent.
		super()._parse(engine)

		# Parse self
		if isinstance(self.args, dict):
			if 'size' not in self.args:
				raise ParsingError('Missing key "size" in pooling container.')
			self.size = engine.evaluate(self.args['size'], recursive=True)

			if 'strides' in self.args:
				self.strides = engine.evaluate(self.args['strides'],
					recursive=True)

			if 'type' in self.args:
				self.pooltype = engine.evaluate(self.args['type']).lower()
				if self.pooltype not in Pooling.POOL_TYPES:
					raise ParsingError('Unknown pool type "{}". Pool type can '
						'be one of: {}'.format(
							self.pooltype, ', '.join(Pooling.POOL_TYPES)
						))
		else:
			self.size = engine.evaluate(self.args, recursive=True)

		if self.pooltype is None:
			self.pooltype = Pooling.POOL_TYPES[0]

		if not isinstance(self.size, (list, tuple)):
			self.size = [self.size]
		if not 1 <= len(self.size) <= 3:
			raise ParsingError('Only pooling layers with dimensions 1, 2, '
				'or 3 are supported.')
		for i in range(len(self.size)):
			try:
				self.size[i] = int(self.size[i])
			except ValueError:
				raise ParsingError('All "size" entries must evaluate to '
					'integers. We received this instead: {}'
					.format(self.size[i]))

		if self.strides is not None:
			if not isinstance(self.strides, (list, tuple)):
				try:
					self.strides = int(self.strides)
				except ValueError:
					raise ParsingError('"strides" must evaluate to an '
						'integer or a list of integers.')
				self.strides = [self.strides] * len(self.size)
			else:
				if len(self.strides) != len(self.size):
					raise ParsingError('If "strides" is a list, it must '
						'be the same length as "size".')
				for i in range(len(self.strides)):
					try:
						self.strides[i] = int(self.strides[i])
					except ValueError:
						raise ParsingError('Each element of "strides" '
							'must evaluate to an integer.')
		else:
			self.strides = [1] * len(self.size)

	###########################################################################
	def _build(self, model):
		""" Instantiates the layer with the given backend.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error

			kwargs = {
				'pool_size' : self.size,
				'strides' : self.strides,
				'border_mode' : 'valid',
				'name' : self.name
			}

			if self.pooltype == 'max':
				func = {
					1 : L.MaxPooling1D,
					2 : L.MaxPooling2D,
					3 : L.MaxPooling3D
				}.get(len(self.size))
			elif self.pooltype == 'average':
				func = {
					1 : L.AveragePooling1D,
					2 : L.AveragePooling2D,
					3 : L.AveragePooling3D
				}.get(len(self.size))
			else:
				raise ValueError('Unhandled pool type "{}". This is a bug.',
					self.pooltype)

			if len(self.size) == 1:
				kwargs['pool_length'] = kwargs.pop('pool_size')
				kwargs['stride'] = kwargs.pop('strides')

			if func is None:
				raise ValueError('Invalid pool function for pool type "{}" '
					'the supplied pool parameters. This is a bug.'
					.format(self.pooltype))

			yield func(**kwargs)

		else:
			raise ValueError(
				'Unknown or unsupported backend: {}'.format(backend))

	###########################################################################
	def shape(self, input_shapes):
		""" Returns the output shape of this layer for a given input shape.
		"""
		if len(input_shapes) > 1:
			raise ValueError('Pooling layers only take a single input.')
		input_shape = input_shapes[0]
		if len(input_shape) != len(self.size) + 1:
			raise ValueError('Invalid input shape to a pooling layer: {}'
				.format(input_shape))

		output_shape = tuple(
			(input_shape[i] + self.strides[i] - 1) // self.strides[i]
			for i in range(len(self.size))
		) + (input_shape[-1], )
		return output_shape

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

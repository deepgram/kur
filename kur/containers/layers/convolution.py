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
class Convolution(Layer):				# pylint: disable=too-few-public-methods
	""" A vanilla convolution layer.

		# Properties

		kernels: int (required). The number of kernels (also called filters) to
			use.
		size: int or list of ints (required). The size of each kernel. If an
			integer is used, it is interpretted as a one-dimensional
			convolution, the same as if it were put into a length-1 list.
		strides: int or list of ints (optional; default: 1 in each dimension).
			The stride (subsampling rate) between convolutions. If a list, it
			must be the same length as `size` and specify the stride in each
			respective dimension. If a single number, it is used as the stride
			in each dimension.
		border: "valid" or "same" (default: "same").

		# Example

		```
		convolution:
		  kernels: 64
		  size: [2, 2]
		  strides: 1
		  border: relu
		```
	"""

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new convolution layer.
		"""
		super().__init__(*args, **kwargs)
		self.kernels = None
		self.size = None
		self.strides = None
		self.activation = None
		self.border = None

	###########################################################################
	def _parse(self, engine):
		""" Parses out the convolution layer.
		"""
		# Always call the parent.
		super()._parse(engine)

		# Parse self
		if 'kernels' not in self.args:
			raise ParsingError('Missing key "kernels" in convolution '
				'container.')
		self.kernels = engine.evaluate(self.args['kernels'])
		try:
			self.kernels = int(self.kernels)
		except ValueError:
			raise ParsingError('"kernels" must evaluate to an integer.')

		if 'size' not in self.args:
			raise ParsingError('Missing key "size" in convolution container.')
		self.size = engine.evaluate(self.args['size'], recursive=True)
		if not isinstance(self.size, (list, tuple)):
			self.size = [self.size]
		if not 1 <= len(self.size) <= 3:
			raise ParsingError('Only convolutions with dimensions 1, 2, or 3 '
				'are supported.')
		for i in range(len(self.size)):
			try:
				self.size[i] = int(self.size[i])
			except ValueError:
				raise ParsingError('All "size" entries must evaluate to '
					'integers. We received this instead: {}'
					.format(self.size[i]))

		if 'strides' in self.args:
			self.strides = engine.evaluate(self.args['strides'], recursive=True)
			if not isinstance(self.strides, (list, tuple)):
				try:
					self.strides = int(self.strides)
				except ValueError:
					raise ParsingError('"strides" must evaluate to an integer '
						'or a list of integers.')
				self.strides = [self.strides] * len(self.size)
			else:
				if len(self.strides) != len(self.size):
					raise ParsingError('If "strides" is a list, it must be the '
						'same length as "size".')
				for i in range(len(self.strides)):
					try:
						self.strides[i] = int(self.strides[i])
					except ValueError:
						raise ParsingError('Each element of "strides" must '
							'evaluate to an integer.')
		else:
			self.strides = [1] * len(self.size)

		self.activation = None

		if 'border' in self.args:
			self.border = engine.evaluate(self.args['border'])
			if not isinstance(self.border, str) or \
				not self.border in ('valid', 'same'):
				raise ParsingError('"border" must be one of: "valid", "same".')
		else:
			self.border = 'same'

	###########################################################################
	def _build(self, model):
		""" Instantiates the layer with the given backend.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error

			kwargs = {
				'nb_filter' : self.kernels,
				'activation' : self.activation or 'linear',
				'border_mode' : self.border,
				'name' : self.name
			}

			if len(self.size) == 1:
				func = L.Convolution1D
				kwargs.update({
					'filter_length' : self.size[0],
					'subsample_length' : self.strides[0]
				})
			elif len(self.size) == 2:
				func = L.Convolution2D
				kwargs.update({
					'nb_row' : self.size[0],
					'nb_col' : self.size[1],
					'subsample' : self.strides
				})
			elif len(self.size) == 3:
				func = L.Convolution3D
				kwargs.update({
					'kernel_dim1' : self.size[0],
					'kernel_dim2' : self.size[1],
					'kernel_dim3' : self.size[2],
					'subsample' : self.strides
				})
			else:
				raise ValueError('Unhandled convolution dimension: {}. This is '
					'a bug.'.format(len(self.size)))

			yield func(**kwargs)

		else:
			raise ValueError(
				'Unknown or unsupported backend: {}'.format(backend))

	###########################################################################
	def shape(self, input_shapes):
		""" Returns the output shape of this layer for a given input shape.
		"""
		if len(input_shapes) > 1:
			raise ValueError('Convolutions only take a single input.')
		input_shape = input_shapes[0]
		if len(input_shape) != len(self.size) + 1:
			raise ValueError('Invalid input shape to a convolution layer: {}'
				.format(input_shape))

		def apply_border(input_shape, size):
			if self.border == 'same':
				return input_shape
			else:
				return input_shape - size + 1

		output_shape = tuple(
			(apply_border(input_shape[i], self.size[i]) \
				+ self.strides[i] - 1) // self.strides[i]
			for i in range(len(self.size))
		) + (self.kernels, )
		return output_shape

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

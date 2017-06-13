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

import logging

from . import Layer, ParsingError

logger = logging.getLogger(__name__)

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

	SUPPORTED_TYPES = ('standard', 'highway')

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
		self.type = None
		self.highway_bias = None

	###########################################################################
	def _parse(self, engine):
		""" Parses out the convolution layer.
		"""
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

		assert self.border in ('same', 'valid')

		if 'type' in self.args:
			self.type = engine.evaluate(self.args['type'])
			if not isinstance(self.type, str) or \
				not self.type in self.SUPPORTED_TYPES:
				raise ParsingError('"type" must be one of: {}'.format(
					', '.join(self.SUPPORTED_TYPES)))
		else:
			self.type = self.SUPPORTED_TYPES[0]

		assert self.type in self.SUPPORTED_TYPES

		if 'bias' in self.args:
			self.highway_bias = engine.evaluate(self.args['bias'])
			try:
				self.highway_bias = float(self.highway_bias)
			except ValueError:
				raise ParsingError('"bias" term must be a floating-point '
					'number. Received: {}'.format(self.highway_bias))

			if self.type != 'highway':
				logger.warning('"bias" term was specified, but is only used '
					'with "highway"-type convolutions. Ignoring this...')
		else:
			self.highway_bias = -1.

	###########################################################################
	def _build(self, model):
		""" Instantiates the layer with the given backend.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			if self.type != 'standard':
				raise ValueError('Backend does not support the requested CNN '
					'type: {}'.format(self.type))

			if backend.keras_version() == 1:
				import keras.layers as L			# pylint: disable=import-error

				kwargs = {
					'nb_filter' : self.kernels,
					'activation' : self.activation or 'linear',
					'border_mode' : self.border,
					'name' : self.name,
					'trainable' : not self.frozen
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
					raise ValueError('Unhandled convolution dimension: {}. This '
						'is a bug.'.format(len(self.size)))

				yield func(**kwargs)

			else:

				import keras.layers.convolutional as L # pylint: disable=import-error

				kwargs = {
					'filters' : self.kernels,
					'activation' : self.activation or 'linear',
					'padding' : self.border,
					'name' : self.name,
					'kernel_size' : self.size,
					'strides' : self.strides,
					'trainable' : not self.frozen
				}

				if len(self.size) == 1:
					func = L.Conv1D
				elif len(self.size) == 2:
					func = L.Conv2D
				elif len(self.size) == 3:
					func = L.Conv3D
				else:
					raise ValueError('Unhandled convolution dimension: {}. This '
						'is a bug.'.format(len(self.size)))

				if len(self.size) > 1:
					kwargs['data_format'] = 'channels_last'

				yield func(**kwargs)

		elif backend.get_name() == 'pytorch':

			# pylint: disable=import-error
			import torch.nn as nn
			import torch.nn.functional as F
			# pylint: enable=import-error

			from kur.backend.pytorch.modules import swap_channels, multiply, \
				add, constant_minus

			func = {
				1 : nn.Conv1d,
				2 : nn.Conv2d,
				3 : nn.Conv3d
			}.get(len(self.size))
			if not func:
				raise ValueError('Unhandled convolution dimension: {}. This '
					'is a bug.'.format(len(self.size)))

			if self.border == 'valid':
				padding = 0
			else:
				# "same" padding requires you to pad with P = S - 1 zeros
				# total. However, PyTorch always pads both sides of the input
				# tensor, implying that PyTorch only accepts padding P' such
				# that P = 2P'. This unfortunately means that if S is even,
				# then the desired padding P is odd, and so no P' exists.
				if any(s % 2 == 0 for s in self.size):
					raise ValueError('PyTorch convolutions cannot use "same" '
						'border mode when the receptive field "size" is even.')
				padding = tuple((s-1)//2 for s in self.size)

			def layer(in_channels, bias=None):
				result = func(
					in_channels,
					self.kernels,
					tuple(self.size),
					stride=tuple(self.strides),
					padding=padding
				)
				if bias is not None:
					result.bias.requires_grad = False
					result.bias.data.fill_(bias)
				return result

			if self.type == 'standard':

				def connect(inputs):
					""" Connects the layer.
					"""
					assert len(inputs) == 1
					output = model.data.add_operation(
						swap_channels.begin
					)(inputs[0]['layer'])
					output = model.data.add_layer(
						self.name,
						layer(inputs[0]['shape'][-1]),
						frozen=self.frozen
					)(output)
					output = model.data.add_operation(
						swap_channels.end
					)(output)
					return {
						'shape' : self.shape([inputs[0]['shape']]),
						'layer' : output
					}

			elif self.type == 'highway':

				def connect(inputs):
					""" Connects the layer.
					"""
					assert len(inputs) == 1

					H = model.data.add_layer(
						self.name + '_H',
						layer(inputs[0]['shape'][-1]),
						frozen=self.frozen
					).op

					T = model.data.add_layer(
						self.name + '_T',
						layer(
							inputs[0]['shape'][-1],
							bias=self.highway_bias
						),
						frozen=True if self.frozen else None
					).op

					def func(module, x):
						h = H(module, x)
						h = F.relu(h)
						t = T(module, x)
						t = F.sigmoid(t)
						return h*t + x*(1-t)
					func.pure = True

					output = model.data.add_operation(
						swap_channels.begin
					)(inputs[0]['layer'])
					output = model.data.add_operation(func)(output)
					output = model.data.add_operation(
						swap_channels.end
					)(output)
					return {
						'shape' : self.shape([inputs[0]['shape']]),
						'layer' : output
					}

			else:
				raise ValueError('Unhandled CNN type: {}. This is a bug.'
					.format(self.type))

			yield connect

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
			""" Resolves output shape differences caused by border mode.
			"""
			if self.border == 'same':
				return input_shape
			return input_shape - size + 1

		output_shape = tuple(
			(apply_border(input_shape[i], self.size[i]) + self.strides[i] - 1) \
				// self.strides[i] if input_shape[i] else None
			for i in range(len(self.size))
		) + (self.kernels, )
		return output_shape

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

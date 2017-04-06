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
class BatchNormalization(Layer):	# pylint: disable=too-few-public-methods
	""" A batch normalization layer, which normalizes activations.
	"""

	###########################################################################
	@classmethod
	def get_container_name(cls):
		""" Returns the name of the container class.
		"""
		return 'batch_normalization'

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new batch normalization layer.
		"""
		super().__init__(*args, **kwargs)
		self.axis = None
		self.momentum = 0.

	###########################################################################
	def _parse(self, engine):
		""" Parse the layer.
		"""
		if isinstance(self.args, dict):
			if 'axis' in self.args:
				self.axis = engine.evaluate(self.args['axis'], recursive=True)
				if not isinstance(self.axis, int):
					raise ParsingError('"axis" must be an integer.')
			if 'momentum' in self.args:
				self.momentum = engine.evaluate(self.args['momentum'],
					recursive=True)
				if not isinstance(self.momentum, (int, float)):
					raise ParsingError('"momentum" must be numeric.')

	###########################################################################
	def _build(self, model):
		""" Instantiates the layer with the given backend.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			if backend.keras_version() == 1:
				import keras.layers as L		# pylint: disable=import-error
				yield L.BatchNormalization(
					mode=2,
					axis=-1 if self.axis is None else self.axis,
					name=self.name,
					trainable=not self.frozen
				)
			else:
				import keras.layers.normalization as L # pylint: disable=import-error

				###############################################################
				# pylint: disable=too-few-public-methods,unused-argument
				class CustomBatchNormalization(L.BatchNormalization):
					""" Custom batch-normalization implementation
					"""
					def call(self, inputs, training=None):
						""" Forces Keras to respect the way we want batch
							normalization to be calculated.
						"""
						return super().call(inputs, training=False)
				# pylint: enable=too-few-public-methods,unused-argument

				yield CustomBatchNormalization(
					axis=-1 if self.axis is None else self.axis,
					center=True,
					scale=True,
					name=self.name,
					momentum=self.momentum,
					trainable=not self.frozen
				)

		elif backend.get_name() == 'pytorch':

			import torch.nn as nn				# pylint: disable=import-error
			from kur.backend.pytorch.modules import swap_channels

			def connect(inputs):
				""" Connects the layer
				"""
				assert len(inputs) == 1

				ndim = len(inputs[0]['shape'])
				func = {
					0 : nn.BatchNorm1d,
					1 : nn.BatchNorm1d,
					2 : nn.BatchNorm2d,
					3 : nn.BatchNorm3d
				}.get(ndim-1)
				if func is None:
					raise ValueError('Unsupported dimensionality: {}'
						.format(ndim))

				output = inputs[0]['layer']

				if ndim != 1:
					output = model.data.add_operation(
						swap_channels.begin
					)(output)

				output = model.data.add_layer(
					self.name,
					func(
						inputs[0]['shape'][-1],
						affine=True,
						momentum=self.momentum
					),
					frozen=self.frozen
				)(output)

				if ndim != 1:
					output = model.data.add_operation(
						swap_channels.end
					)(output)

				return {
					'shape' : self.shape([inputs[0]['shape']]),
					'layer' : output
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
			raise ValueError('Batch normalizations only take a single input.')
		input_shape = input_shapes[0]
		return input_shape

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

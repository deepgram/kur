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
class Dense(Layer):						# pylint: disable=too-few-public-methods
	""" A fully-connected layer.
	"""

	DEFAULT_AUTO_FLATTEN = False

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new dense layer.
		"""
		super().__init__(*args, **kwargs)
		self.size = None
		self.auto_flatten = None

	###########################################################################
	def _parse(self, engine):
		""" Parse the layer.
		"""

		super()._parse(engine)

		if isinstance(self.args, dict):
			self.size = engine.evaluate(self.args['size'], recursive=True)
			self.auto_flatten = engine.evaluate(
				self.args.get('flatten', Dense.DEFAULT_AUTO_FLATTEN),
				recursive=True
			)
		elif isinstance(self.args, list):
			self.size = engine.evaluate(self.args, recursive=True)
			self.auto_flatten = Dense.DEFAULT_AUTO_FLATTEN
		else:
			self.size = self.args
			self.auto_flatten = Dense.DEFAULT_AUTO_FLATTEN

		if not isinstance(self.size, (tuple, list)):
			self.size = [self.size]

		try:
			for i, v in enumerate(self.size):
				self.size[i] = int(v)
		except ValueError:
			raise ParsingError('Key "size" in Dense layer must be an integer '
				'or a list of integers. Received: {}'.format(self.size))

		if not isinstance(self.auto_flatten, bool):
			raise ParsingError('"auto_flatten" key in Dense layer must be '
				'boolean. Received: {}'.format(self.auto_flatten))

	###########################################################################
	def _build(self, model):
		""" Create the backend-specific placeholder.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error

			if self.auto_flatten:
				yield L.Flatten()

			if backend.keras_version() == 1:
				func = lambda x, **kwargs: L.Dense(output_dim=x, **kwargs)
			else:
				func = lambda x, **kwargs: L.Dense(units=x, **kwargs)

			for v in self.size[:-1]:
				yield func(v)

			yield func(
				self.size[-1],
				name=self.name
			)

		elif backend.get_name() == 'pytorch':

			import torch.nn as nn				# pylint: disable=import-error

			def connect(inputs):
				""" Connects the layer.
				"""
				assert len(inputs) == 1
				return {
					'shape' : self.shape([inputs[0]['shape']]),
					'layer' : model.data.add_layer(
						self.name,
						nn.Linear(inputs[0]['shape'][-1], self.size[-1])
					)(inputs[0]['layer'])
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
			raise ValueError('Dense layers only take a single input.')
		return (self.size[-1], )

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

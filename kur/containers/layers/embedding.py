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
class Embedding(Layer):				# pylint: disable=too-few-public-methods
	""" A layer maps integer labels to a higher-dimensional embedding.
	"""

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new embedding layer.
		"""
		super().__init__(*args, **kwargs)
		self.vocab_size = None
		self.size = None

	###########################################################################
	def _parse(self, engine):
		""" Parse the layer.
		"""
		if not isinstance(self.args, dict):
			raise ParsingError('Embedding layer expected a dictionary of '
				'parameters, but instead received: {}'.format(self.args))

		if 'vocab_size' not in self.args:
			raise ParsingError('Missing "vocab_size" key in Embedding layer.')
		self.vocab_size = engine.evaluate(self.args['vocab_size'],
			recursive=True)
		try:
			self.vocab_size = int(self.vocab_size)
		except ValueError:
			raise ParsingError('Key "vocab_size" in Embedding layer must be '
				'an integer.')
		if self.vocab_size < 1:
			raise ParsingError('Key "vocab_size" in Embedding layer must be '
				'>= 1.')

		if 'size' not in self.args:
			raise ParsingError('Missing "size" key in Embedding layer.')
		self.size = engine.evaluate(self.args['size'], recursive=True)
		try:
			self.size = int(self.size)
		except ValueError:
			raise ParsingError('Key "size" in Embedding layer must be an '
				'integer.')
		if self.size < 1:
			raise ParsingError('Key "size" in Embedding layer must be >= 1.')

	###########################################################################
	def _build(self, model):
		""" Create the backend-specific placeholder.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error

			yield L.Embedding(
				self.vocab_size,
				self.size,
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
						lambda *x: nn.Embedding(
							self.vocab_size,
							self.size
						)(*[X.long() for X in x])
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
			raise ValueError('Embedding layers only take a single input.')
		input_shape = input_shapes[0]
		if len(input_shape) != 1:
			raise ValueError('Embedding layers only accept flat (1D) inputs.')
		return input_shape + (self.size, )

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

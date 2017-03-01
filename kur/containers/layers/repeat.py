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
class Repeat(Layer):				# pylint: disable=too-few-public-methods
	""" A layer which repeats its input a fixed number of times.
	"""

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new repeat layer.
		"""
		super().__init__(*args, **kwargs)
		self.count = None

	###########################################################################
	def _parse(self, engine):
		""" Parse the layer.
		"""

		super()._parse(engine)

		if isinstance(self.args, dict):
			self.count = engine.evaluate(self.args['count'], recursive=True)
		else:
			self.count = self.args

		try:
			self.count = int(self.count)
		except ValueError:
			raise ParsingError('Key "count" in Repeat layer must be an '
				'integer. Received: {}'.format(self.count))

		if self.count < 1:
			raise ParsingError('Key "count" in Repeat layer must be >= 1.')

	###########################################################################
	def _build(self, model):
		""" Create the backend-specific placeholder.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error

			yield L.RepeatVector(
				self.count,
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
			raise ValueError('Repeat layers only take a single input.')
		input_shape = input_shapes[0]
		if len(input_shape) != 1:
			raise ValueError('Repeat layers only accept flat (1D) inputs.')
		return (self.count, input_shape[0])

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

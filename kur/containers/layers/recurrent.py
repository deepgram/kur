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
class Recurrent(Layer):				# pylint: disable=too-few-public-methods
	""" A recurrent neural network.

		# Properties

		sequence: bool.
		size: int.
		bidirectional: bool.
		merge: one of (multiply, add, concat, average)
		type: one of (lstm, gru)

		# Example

		```
		recurrent:
		  size: 32
		  sequence: yes
		  bidirectional: yes
		  merge: average
		  type: lstm
		```
	"""

	MERGE_TYPES = ('multiply', 'add', 'concat', 'average')
	RNN_TYPES = ('lstm', 'gru')

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new recurrent layer.
		"""
		super().__init__(*args, **kwargs)
		self.type = None
		self.size = None
		self.sequence = None
		self.bidirectional = None
		self.merge = None

	###########################################################################
	def _parse(self, engine):
		""" Parses out the recurrent layer.
		"""
		# Always call the parent.
		super()._parse(engine)

		self.sequence = engine.evaluate(self.args.get('sequence', True),
			recursive=True)
		if not isinstance(self.sequence, bool):
			raise ParsingError('Wrong type for "sequence" argument in '
				'recurrent layer. Expected bool, received: {}'
				.format(self.sequence))

		self.bidirectional = engine.evaluate(
			self.args.get('bidirectional', False),
			recursive=True
		)
		if not isinstance(self.bidirectional, bool):
			raise ParsingError('Wrong type for "bidirectional" argument in '
				'recurrent layer. Expected bool, received: {}'
				.format(self.bidirectional))

		self.merge = engine.evaluate(self.args.get('merge'), recursive=True)
		if not self.bidirectional:
			if self.merge is not None:
				raise ParsingError('Having a "merge" strategy in a '
					'"recurrent" layer only makes sense for bidirectional '
					'RNNs.')
		else:
			if self.merge is None:
				self.merge = 'average'
			if not isinstance(self.merge, str):
				raise ParsingError('Wrong type for "merge" argument in '
					'recurrent layer. Expected one of: {}. Received: {}'
						.format(', '.join(Recurrent.MERGE_TYPES), self.merge)
					)
			self.merge = self.merge.lower()
			if self.merge not in Recurrent.MERGE_TYPES:
				raise ParsingError('Bad value for "merge" argument in '
					'recurrent layer. Expected one of: {}. Received: {}'
						.format(', '.join(Recurrent.MERGE_TYPES), self.merge)
					)

		self.type = engine.evaluate(self.args.get('type', 'gru'),
			recursive=True)
		if not isinstance(self.type, str):
			raise ParsingError('Wrong type for "type" argument in recurrent '
				'layer. Expected one of: {}. Received: {}'.format(
					', '.join(Recurrent.RNN_TYPES), self.type
				))
		self.type = self.type.lower()
		if self.type not in Recurrent.RNN_TYPES:
			raise ParsingError('Bad value for "type" argument in recurrent '
				'layer. Expected one of: {}. Received: {}'.format(
					', '.join(Recurrent.RNN_TYPES), self.type
				))

		self.size = engine.evaluate(self.args.get('size'), recursive=True)
		if not isinstance(self.size, int):
			raise ParsingError('Bad or missing value for "size" argument in '
				'recurrent layer. Expected an integer. Received: {}'
				.format(self.size))

		if 'outer_activation' in self.args:
			self.activation = engine.evaluate(self.args['outer_activation'])
		else:
			self.activation = None

	###########################################################################
	def _build(self, model):
		""" Instantiates the layer with the given backend.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error

			func = {
				'lstm' : L.LSTM,
				'gru' : L.GRU
			}.get(self.type)
			if func is None:
				raise ValueError('Unhandled RNN type: {}. This is a bug.'
					.format(self.type))

			kwargs = {
				'activation' : self.activation or 'relu',
				'output_dim' : self.size,
				'return_sequences' : self.sequence,
				'go_backwards' : False
			}

			if self.bidirectional:
				kwargs['name'] = self.name + '_fwd'

				if self.merge in ('concat', ):
					if kwargs['output_dim'] % 2 != 0:
						logger.warning('Recurrent layer "%s" has an odd '
							'number for "size", but has a concat-type merge '
							'strategy. We are going to reduce its size by '
							'one.', self.name)
						kwargs['output_dim'] -= 1
					kwargs['output_dim'] //= 2

				forward = func(**kwargs)

				kwargs['go_backwards'] = True
				kwargs['name'] = self.name + '_rev'
				backward = func(**kwargs)

				def merge(tensor):
					""" Returns a bidirectional RNN.
					"""
					return L.merge(
						[forward(tensor), backward(tensor)],
						mode={
							'multiply' : 'mul',
							'add' : 'sum',
							'concat' : 'concat',
							'average' : 'ave'
						}.get(self.merge),
						name=self.name,
						**{
							'concat' : {'concat_axis' : -1}
						}.get(self.merge, {})
					)

				yield merge
			else:
				kwargs['name'] = self.name
				yield func(**kwargs)

		else:
			raise ValueError(
				'Unknown or unsupported backend: {}'.format(backend))

	###########################################################################
	def shape(self, input_shapes):
		""" Returns the output shape of this layer for a given input shape.
		"""
		if len(input_shapes) > 1:
			raise ValueError('Recurrent layers only take a single input.')
		input_shape = input_shapes[0]
		return input_shape[:-1] + (self.size, )

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

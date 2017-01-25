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
class Merge(Layer):					# pylint: disable=too-few-public-methods
	""" A container for merging inputs from multiple input layers.
	"""

	MERGE_MODES = ('multiply', 'add', 'concat', 'average')
	DEFAULT_MERGE_MODE = 'average'

	DEFAULT_AXIS = -1

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Create a new merge container.
		"""
		super().__init__(*args, **kwargs)
		self.mode = None
		self.axis = None

	###########################################################################
	def _parse(self, engine):
		""" Parse the child containers
		"""

		# Always call the parent.
		super()._parse(engine)

		if isinstance(self.args, dict):
			self.mode = engine.evaluate(self.args.get('mode'), recursive=True)
			self.axis = engine.evaluate(self.args.get('axis'), recursive=True)
		else:
			self.mode = self.args
			self.axis = Merge.DEFAULT_AXIS

		if self.mode is None:
			self.mode = Merge.DEFAULT_MERGE_MODE
		if not isinstance(self.mode, str):
			raise ParsingError('Wrong type for "mode" argument in '
				'recurrent layer. Expected one of: {}. Received: {}'
					.format(', '.join(Merge.MERGE_MODES), self.mode)
				)
		self.mode = self.mode.lower()
		if self.mode not in Merge.MERGE_MODES:
			raise ParsingError('Bad value for "mode" argument in '
				'recurrent layer. Expected one of: {}. Received: {}'
					.format(', '.join(Merge.MERGE_MODES), self.mode)
				)

		if not isinstance(self.axis, int):
			raise ParsingError('Axis must be an integer. Received: {}'
				.format(self.axis))

	###########################################################################
	def _build(self, model):
		""" Instantiate the container.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error

			def merge(inputs):
				""" Merges inputs.
				"""
				# Keras "merge" requires more than one layer.
				if len(inputs) == 1:
					return inputs[0]
				return L.merge(
					inputs,
					mode={
						'multiply' : 'mul',
						'add' : 'sum',
						'concat' : 'concat',
						'average' : 'ave'
					}.get(self.mode),
					name=self.name,
					**{
						'concat' : {'concat_axis' : self.axis}
					}.get(self.mode, {})
				)

			yield merge

		else:
			raise ValueError(
				'Unknown or unsupported backend: {}'.format(backend)
			)

	###########################################################################
	def shape(self, input_shapes):
		""" Returns the output shape of this layer for a given input shape.
		"""
		if self.mode == 'concat':
			result = None
			for input_shape in input_shapes:
				if result is None:
					result = list(input_shape)
				else:
					result[self.axis] += input_shape[self.axis]
			return tuple(result)

		return input_shapes[0]

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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

################################################################################
class Dense(Layer):						# pylint: disable=too-few-public-methods
	""" A fully-connected layer.
	"""

	############################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new dense layer.
		"""
		super().__init__(*args, **kwargs)
		self.size = None

	############################################################################
	def _parse(self, engine):
		""" Parse the layer.
		"""

		super()._parse(engine)

		if isinstance(self.args, dict):
			self.size = engine.evaluate(self.args['size'], recursive=True)
		elif isinstance(self.args, list):
			self.size = engine.evaluate(self.args, recursive=True)
		else:
			self.size = self.args

		if not isinstance(self.size, (tuple, list)):
			self.size = [self.size]

		try:
			for i, v in enumerate(self.size):
				self.size[i] = int(v)
		except ValueError:
			raise ParsingError('Key "size" in Dense layer must be an integer '
				'or a list of integers. Received: {}'.format(self.size))

	############################################################################
	def _build(self, backend):
		""" Create the backend-specific placeholder.
		"""
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error

			for v in self.size[:-1]:
				yield L.Dense(output_dim=v)

			yield L.Dense(
				output_dim=self.size[-1],
				name=self.name
			)

		else:
			raise ValueError(
				'Unknown or unsupported backend: {}'.format(backend))

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

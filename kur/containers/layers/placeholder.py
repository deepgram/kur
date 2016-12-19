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
import warnings
from . import Layer, ParsingError

logger = logging.getLogger(__name__)

###############################################################################
class Placeholder(Layer):				# pylint: disable=too-few-public-methods
	""" A shape placeholder which serves as a size hint, typically used as the
		first layer for each input to the model.

		Placeholders are also convenient places to name an input stream.
	"""

	###########################################################################
	@classmethod
	def get_container_name(cls):
		""" Returns the name of the container class.
		"""
		return 'input'

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new placeholder.
		"""
		super().__init__(*args, **kwargs)
		self.shape = None

	###########################################################################
	def _parse_pre(self, engine):
		""" Pre-parsing hook.
		"""
		super()._parse_pre(engine)

		container_name = self.get_container_name()
		if container_name in self.data:
			data = engine.evaluate(self.data[container_name])
			if isinstance(data, str):
				if 'name' in self.data:
					name = engine.evaluate(self.data['name'])
					logger.warning('Conflicting naming schemes for '
						'placeholder: "%s" and "%s". Using: "%s".',
						data, name, name)
				else:
					logger.debug('Using short-hand name for placeholder: %s',
						data)
					self.name = data
		else:
			warnings.warn('Parsing oddity in placeholder: {}'.format(
				self.data
			))

	###########################################################################
	def set_shape(self, shape):
		""" Sets a shape.
		"""
		if self.shape is not None:
			logger.warning('Modifying the shape of Placeholder "%s".',
				self.name)

		if not isinstance(shape, (list, tuple)):
			shape = (shape, )
		for x in shape:
			if not isinstance(x, int):
				raise ParsingError(
					'All entries in "shape" must be integers. Shape is: {}'
					.format(shape)
				)
		self.shape = shape

	###########################################################################
	def _parse(self, engine):
		""" Parse the placeholder.
		"""

		super()._parse(engine)

		if 'shape' not in self.args:
			logger.debug('Placeholder "%s" has a deferred shape.', self.name)
		else:
			self.set_shape(engine.evaluate(self.args['shape'], recursive=True))

	###########################################################################
	def _build(self, model):
		""" Create the backend-specific placeholder.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			if self.shape is None:
				raise ParsingError(
					'Placeholder "{}" requires a shape.'.format(self.name))

			import keras.layers as L			# pylint: disable=import-error
			yield L.Input(
				shape=self.shape,
				name=self.name
			)

		else:
			raise ValueError('Unknown or unsupported backend: {}'.format(backend))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

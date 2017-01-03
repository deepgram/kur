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
from . import Layer

logger = logging.getLogger(__name__)

###############################################################################
class Output(Layer):				# pylint: disable=too-few-public-methods
	""" A named dummy layer for outputs.
	"""

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new placeholder.
		"""
		super().__init__(*args, **kwargs)

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
						'output layer: "%s" and "%s". Using: "%s".',
						data, name, name)
				else:
					logger.debug('Using short-hand name for output: %s', data)
					self.name = data
		else:
			warnings.warn('Parsing oddity in output: {}'.format(self.data))

	###########################################################################
	def _parse_post(self, engine):
		""" Post-parsing hook.
		"""
		self.sink = True
		super()._parse_post(engine)

	###########################################################################
	def _build(self, model):
		""" Create the backend-specific placeholder.
		"""
		yield None
		"""backend = model.get_backend()
		if backend.get_name() == 'keras':
			yield lambda inputs: inputs[0]
		else:
			raise ValueError(
				'Unknown or unsupported backend: {}'.format(backend))"""

	###########################################################################
	def shape(self, input_shapes):
		""" Returns the output shape of this layer for a given input shape.
		"""
		if len(input_shapes) > 1:
			raise ValueError('Outputs only take a single input.')
		input_shape = input_shapes[0]
		return input_shape

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

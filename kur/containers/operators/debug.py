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
from . import Operator

logger = logging.getLogger(__name__)

###############################################################################
class Debug(Operator):					# pylint: disable=too-few-public-methods
	""" Simple debug message printer.

		# Example

		Simple debug string:
		```
		debug: Hello World
		```

		With variable substitution:
		```
		debug: "There are {{ layers|length }} layers so far."
		```
	"""

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Create a new debug container.
		"""
		super().__init__(*args, **kwargs)
		self.message = None

	###########################################################################
	def _parse(self, engine):
		""" Parse the debug statement and print it.
		"""
		super()._parse(engine)
		logger.debug('Container %s says: %s', self.name, self.args)

	###########################################################################
	def _build(self, model):	# pylint: disable=unused-argument,no-self-use
		""" Debug statements don't produce layers.
		"""
		return iter(())

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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

import yaml
from . import Reader

###############################################################################
class YamlReader(Reader):
	""" A YAML reader.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the reader.
		"""
		return 'yaml'

	###########################################################################
	@classmethod
	def supported_filetypes(cls):
		""" Returns a list of supported file extensions.
		"""
		return ('yml', 'yaml')

	###########################################################################
	def read(self, data):						# pylint: disable=no-self-use
		""" Reads the data and returns a native Python dictionary.

			# Arguments

			data: str. The data string to parse.

			# Return value

			A Python dictionary or list representing the data.

			# Exceptions

			If there is a YAML syntax/parsing error, a `yaml.error.YAMLError`
			instance is raised.
		"""
		return yaml.load(data)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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

import json
from . import Reader

###############################################################################
class JsonReader(Reader):
	""" A JSON reader.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the reader.
		"""
		return 'json'

	###########################################################################
	@classmethod
	def supported_filetypes(cls):
		""" Returns a list of supported file extensions.
		"""
		return ('json',)

	###########################################################################
	def read(self, data):						# pylint: disable=no-self-use
		""" Reads the data and returns a native Python dictionary.

			# Arguments

			data: str. The data string to parse.

			# Return value

			A Python dictionary or list representing the data.

			# Exceptions

			If there is a JSON syntax/parsing error, a
			`json.decoder.JSONDecodeError` instance is raised.
		"""
		return json.loads(data)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

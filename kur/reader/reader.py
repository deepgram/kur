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

import os
from kur.utils import get_subclasses

###############################################################################
class Reader:
	""" Base class for all readers.

		Readers are responsible for reading files and constructing parsable
		Python objects out of the encoded data. For an example, you might have
		a JSON reader, a YAML reader, etc.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the reader.

			# Return value

			A lower-case string unique to this reader.
		"""
		return cls.__name__.lower()

	###########################################################################
	@staticmethod
	def get_all_readers():
		""" Returns all Reader subclasses.
		"""
		for cls in get_subclasses(Reader):
			yield cls

	###########################################################################
	@staticmethod
	def get_reader_by_name(name):
		""" Finds a reader class with the given name.
		"""
		name = name.lower()
		for cls in Reader.get_all_readers():
			if cls.get_name() == name:
				return cls
		raise ValueError('No such reader with name "{}"'.format(name))

	###########################################################################
	@classmethod
	def supported_filetypes(cls):
		""" Returns a list of supported file extensions.

			# Return value

			A tuple of lowercase extensions (without the period) that generally
			indicate that the Reader subclass can handle the implied file type.
			If no file extensions is particularly indicative, then an empty
			tuple can be returned. If the absence of a file extension is
			particularly indicative, then an empty string can be present in the
			tuple.
		"""
		raise NotImplementedError

	###########################################################################
	@staticmethod
	def get_reader_for_file(filename):
		""" Returns the first Reader that claims to be able to read the given
			file.

			# Arguments

			filename: str. The filename to find a Reader for.

			# Return value

			A class that can read the given filename. If no such class can be
			found, a ValueError is raised.
		"""
		_, ext = os.path.splitext(filename)
		ext = ext.lower()
		if ext.startswith('.'):
			ext = ext[1:]

		for cls in Reader.get_all_readers():
			if ext in cls.supported_filetypes():
				return cls

		raise ValueError(
			'No such reader could be found for file: {}'.format(filename))

	###########################################################################
	@staticmethod
	def read_file(filename):
		""" Convenience function for reading data from a file.
		"""
		reader = Reader.get_reader_for_file(filename)()
		with open(filename) as fh:
			return reader.read(fh.read())

	###########################################################################
	def read(self, data):
		""" Reads the data and returns a native Python dictionary.

			# Arguments

			data: str. The data string to parse.

			# Return value

			A Python dictionary or list representing the data.
		"""
		raise NotImplementedError

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

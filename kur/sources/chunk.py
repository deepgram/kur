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

from . import OriginalSource

###############################################################################
class ChunkSource(OriginalSource):			# pylint: disable=abstract-method
	""" A chunk source is a source which can readily return different lengths
		of data at each iteration.
	"""

	DEFAULT_CHUNK_SIZE = 8192
	USE_BATCH_SIZE = object()
	ALL_ITEMS = object()

	###########################################################################
	@classmethod
	def default_chunk_size(cls):
		""" Returns the default chunk size for this source.
		"""
		return ChunkSource.DEFAULT_CHUNK_SIZE

	###########################################################################
	def __init__(self, chunk_size=None, *args, **kwargs):
		""" Creates a new ChunkSource.
		"""
		super().__init__(*args, **kwargs)
		self.requested_chunk_size = chunk_size or self.default_chunk_size()
		self.chunk_size = self.requested_chunk_size

	###########################################################################
	def set_chunk_size(self, chunk_size):
		""" Modifies the chunk size.
		"""
		self.chunk_size = chunk_size or self.default_chunk_size()

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

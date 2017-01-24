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

import logging
import csv

import numpy

from . import Supplier
from ..sources import VanillaSource
from ..utils import package

logger = logging.getLogger(__name__)

###############################################################################
class CsvSupplier(Supplier):
	""" A supplier which parses a CSV file.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the supplier.
		"""
		return 'csv'

	###########################################################################
	def __init__(self, path=None, url=None, checksum=None, format=None,
		header=None, *args, **kwargs):
		""" Creates a new Pickle supplier.

			# Arguments

			source: str. Filename of pickled file to load.
		"""
		super().__init__(*args, **kwargs)

		target = {
			'path' : path,
			'url' : url,
			'checksum' : checksum,
			'header' : header,
			'format' : format
		}
		self.parse_source({k : v for k, v in target.items() if v is not None})
		self.data = None

	###########################################################################
	def parse_source(self, target):
		""" Parses the data specification.
		"""
		if isinstance(target, str):
			target = {'path' : target}
		path, _ = package.install(
			url=target.get('url'),
			path=target.get('path'),
			checksum=target.get('checksum')
		)

		self.format = target.get('format', {})
		self.header = target.get('header', True)

		self.source = path

	###########################################################################
	def _load(self):
		""" Loads the data (only if it hasn't already been loaded).

			# Note

			By the end of this call, `self.data` will be a dictionary with keys
			for each CSV column, and values which are Source instances for each
			column (e.g., VanillaSource wrapping a numpy array).
		"""
		if self.data is not None:
			return

		logger.info('Reading in CSV file: %s', self.source)
		with open(self.source, newline='') as fh:
			kwargs = {}

			try:
				# Deduce the file format.
				kwargs['dialect'] = csv.Sniffer().sniff(fh.read(1024))
			except csv.Error:
				logger.warning('Failed to sniff data format for file: %s. You '
					'might need to manually choose delimiters, quote '
					'characters, etc.', self.source)
			finally:
				fh.seek(0)

			# Override the format, if necessary.
			if 'delimiter' in self.format:
				kwargs['delimiter'] = self.format['delimiter']
			if 'quote' in self.format:
				kwargs['quotechar'] = self.format['quote']

			# Prepare the reader.
			reader = csv.reader(fh, **kwargs)

			# If we have a header to read, read it now.
			if self.header:
				for row in reader:
					header = row
					break

			# Read the data
			data = numpy.asarray(list(reader), dtype='float32')

			# If we did not have a header, then now is a good time to figure
			# out what our auto-generated header names are going to be.
			if not self.header:
				header = ['column_{}'.format(i) for i in range(len(data[0]))]

		logger.info('Finished reading CSV file.')

		data = data.T
		self.data = {k : VanillaSource(v) for k, v in zip(header, data)}

	###########################################################################
	def get_sources(self, sources=None):
		""" Returns all sources from this provider.
		"""
		self._load()

		if sources is None:
			sources = list(self.data.keys())
		elif not isinstance(sources, (list, tuple)):
			sources = [sources]

		for source in sources:
			if source not in self.data:
				raise KeyError(
					'Invalid data key: {}. Valid keys are: {}'.format(
						source, ', '.join(str(k) for k in self.data)
				))

		return {k : self.data[k] for k in sources}

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

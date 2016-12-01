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

import pickle
import logging
from collections import OrderedDict
import numpy
from . import Supplier
from ..sources import VanillaSource

logger = logging.getLogger(__name__)

###############################################################################
class PickleSupplier(Supplier):
	""" A supplier which parses out data from a Python pickle and exposes it as
		a numpy array. The pickled object is assumed to be a dictionary, with
		keys that map to numpy arrays.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the supplier.
		"""
		return 'pickle'

	###########################################################################
	def __init__(self, source, *args, **kwargs):
		""" Creates a new Pickle supplier.

			# Arguments

			source: str. Filename of pickled file to load.
		"""
		super().__init__(*args, **kwargs)

		self.source = source
		self.data = None

	###########################################################################
	def _load(self):
		""" Loads the data (only if it hasn't already been loaded).
		"""
		if self.data is not None:
			return

		with open(self.source, 'rb') as fh:
			content = fh.read()

		try:
			data = pickle.loads(content)
		except UnicodeDecodeError:
			data = pickle.loads(content, encoding='latin1')
			logger.warning('We needed to explicitly set a "latin1" encoding '
				'to properly load the pickled data. This is probably because '
				'the pickled data was created in Python 2. You really should '
				'switch over to Python 3 in order to ensure future '
				'compatibility.')

		if not isinstance(data, (dict, OrderedDict)):
			raise ValueError('Pickled data must be a dictionary of numpy '
				'arrays.')

		result = {}
		for k, v in data.items():
			if not isinstance(k, str):
				logger.warning('Dictionary keys should be strings. Instead we '
					'received: %s. We are going to try to cast this to a '
					'and keep on going. This might break things. Please use '
					'string keys in the future.', k)
				k = str(k)

			if isinstance(v, (list, tuple)):
				v = numpy.array(v)
			elif not isinstance(v, numpy.ndarray):
				logger.warning('The value corresponding to pickled dictionary '
					'key "%s" is an unexpected type: %s. We are going to try '
					'and wrap this in a numpy array as see what happens. In '
					'future, be sure to store native numpy arrays as the '
					'dictionary keys, or at least try using Python lists or '
					'tuples.', k, type(v))
				v = numpy.array(v)

			if v.dtype.kind == 'O':
				logger.warning('The pickled numpy array we are trying to '
					'process for key "%s" has an underlying "object" data '
					'type. This could be because the array is ragged, or '
					'because the data itself is not numeric. We will try to '
					'use the data, but this could cause unexpected problems.',
					k)

			result[k] = VanillaSource(v)

		self.data = result

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
						source, ', '.join(str(k) for k in self.data.keys())
				))

		return {k : self.data[k] for k in sources}

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

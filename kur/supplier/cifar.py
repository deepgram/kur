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

import tarfile
import os
import re
import pickle

import numpy

from ..utils import package
from . import Supplier
from ..sources import VanillaSource

###############################################################################
class CifarSupplier(Supplier):
	""" A supplier which supplies MNIST image/label pairs. These are downloaded
		from the internet, verified, and parsed as IDX files.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the supplier.
		"""
		return 'cifar'

	###########################################################################
	def __init__(self, url=None, checksum=None, path=None, parts=None,
		*args, **kwargs):
		""" Creates a new CIFAR supplier.

			# Arguments

		"""
		super().__init__(*args, **kwargs)

		path, _ = package.install(
			url=url,
			checksum=checksum,
			path=path
		)
		images, labels = CifarSupplier._load_parts(path, parts)

		self.data = {
			'images' : VanillaSource(CifarSupplier._normalize(images)),
			'labels' : VanillaSource(CifarSupplier._onehot(labels))
		}

	###########################################################################
	@staticmethod
	def _load_parts(path, parts):

		if parts is not None:
			if not isinstance(parts, (list, tuple)):
				parts = [parts]

		result = {}

		with tarfile.open(path, 'r') as tar:
			for member in tar.getmembers():
				_, filename = os.path.split(member.name)

				match = re.match('^data_batch_([0-9]+)$', filename)
				if match:
					val = match.group(1)
				elif filename == 'test_batch':
					val = 'test'
				else:
					continue

				i = val # shut pylint up
				if parts is not None:
					for i in parts:
						if str(i) == val:
							break
					else:
						continue

				if i in result:
					raise ValueError('Too many matches in extracted CIFAR '
						'data for: {}'.format(i))

				stream = tar.extractfile(member)
				if stream is None:
					continue

				content = stream.read()
				data = pickle.loads(content, encoding='latin1')
				result[i] = data

		if parts is None:
			return list(result.values())

		if len(result) != len(parts):
			raise ValueError('Failed to find all pieces of the extract CIFAR '
				'data: {}'.format(parts))

		return (
			numpy.concatenate([result[i]['data'] for i in parts]),
			numpy.concatenate([result[i]['labels'] for i in parts])
		)

	###########################################################################
	@staticmethod
	def _onehot(source):
		onehot = numpy.zeros((source.shape[0], 10))
		for i, row in enumerate(source):
			onehot[i][row] = 1
		return onehot

	###########################################################################
	@staticmethod
	def _normalize(source):
		# Numpy won't automatically promote the uint8 fields to float32.
		data = source.astype(numpy.float32)
		data -= data.mean(axis=0)
		data /= 255
		data = data.reshape((-1, 3, 32, 32))
		data = numpy.transpose(data, axes=(0, 2, 3, 1))
		return data

	###########################################################################
	def get_sources(self, sources=None):
		""" Returns all sources from this provider.
		"""

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

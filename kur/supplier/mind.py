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
import json
import numpy
import scipy
import scipy.ndimage
from . import Supplier
from ..sources import DerivedSource, VanillaSource, ChunkSource
from ..utils import package


class SampleSource(ChunkSource):
	@classmethod
	def default_chunk_size(cls):
		""" Returns the default chunk size for this source.
		"""
		return ChunkSource.USE_BATCH_SIZE

	###########################################################################
	def __init__(self, data, *args, **kwargs):
		super(SampleSource, self).__init__(*args, **kwargs)
		self.data = data
		self.indices = numpy.arange(len(self))

	###########################################################################
	def __iter__(self):
		""" Return an iterator to the data.
		"""
		start = 0
		num_entries = len(self)
		while start < num_entries:
			end = min(num_entries, start + self.chunk_size)
			batch = [self.data[i] for i in self.indices[start:end]]
			yield batch
			start = end

	###########################################################################
	def shape(self):
		""" Return the shape of the tensor (excluding batch size) returned by
			this data source.
		"""
		return (1, )

	###########################################################################
	def can_shuffle(self):
		""" This source can be shuffled.
		"""
		return True

	###########################################################################
	def shuffle(self, indices):
		""" Applies a permutation to the data.
		"""
		if len(indices) > len(self):
			raise ValueError('Shuffleable was asked to apply permutation, but '
				'the permutation is longer than the length of the data set.')
		self.indices[:len(indices)] = self.indices[:len(indices)][indices]

	###########################################################################
	def __len__(self):
		return len(self.data)

class SpectralSource(DerivedSource):
	def __init__(self, source, path, length):
		super().__init__()
		self.source = source
		self.path  = path
		self.length = length

	def derive(self, samples):
		samples = samples[0]
		uuids = [s['uuid'] for s in samples]
		image_paths = [os.path.join(self.path, 'images', '{}.png'.format(u)) for u in uuids]

		derived = numpy.array([scipy.ndimage.imread(p).transpose() for p in image_paths])
		derived = derived.astype(numpy.float32)
		derived /= 255
		return derived
	def shape(self):
		return (864, 192)
	def requires(self):
		return (self.source, )

	def __len__(self):
		return self.length

class CategorySource(DerivedSource):
	def __init__(self, source, length):
		super().__init__()
		self.source = source
		self.length = length

	def derive(self, samples):
		samples = samples[0]
		return numpy.array([[x['category_label']] for x in samples], dtype='int32')
	def shape(self):
		return (1,)
	def requires(self):
		return (self.source, )

	def __len__(self):
		return self.length

class PreciseSource(DerivedSource):
	def __init__(self, source, length):
		super().__init__()
		self.source = source
		self.length = length

	def derive(self, samples): 
		samples = samples[0]
		return numpy.array([[x['category_label']] for x in samples], dtype='int32')
	def shape(self):
		return (1,)
	def requires(self):
		return (self.source, )

	def __len__(self):
		return self.length

class MindSupplier(Supplier):
	"""
	A supplier for mind data!!
	"""
	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the supplier.
		"""
		return 'mind'


	###########################################################################
	def __init__(self, path=None, url=None, checksum=None, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.load_data(path, url, checksum)

		sample_source = SampleSource(self.data)
		length = len(self.data)
		spectral_source = SpectralSource('sample_metadata', path, length)
		category_source = CategorySource('sample_metadata', length)
		precise_source = PreciseSource('sample_metadata', length)
		self.sources = {
			'sample_metadata': sample_source,
			'brain_probe_image': spectral_source,
			'category_label':  category_source,
			'precise_label': precise_source
		}
	
	###########################################################################
	def load_data(self, path, url, checksum):
		
		self.path, _ = package.install(
			url=url,
			path=path,
			checksum=checksum
		)

		jsonl_candidates = [f for f in os.listdir(self.path) if f[-5:] == 'jsonl']
		if len(jsonl_candidates) != 1:
			raise Exception("Supplied path is not a valid data path for MindSupplier")
		
		jsonl_path = os.path.join(self.path, jsonl_candidates[0])
		self.data = []
		with open(jsonl_path) as fin:
			for line in fin:
				self.data.append(json.loads(line))

	###########################################################################
	def get_sources(self):
		return {k: v for k, v in self.sources.items()}


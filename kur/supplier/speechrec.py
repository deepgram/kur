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

import json
import re
import os
import logging
import random
import multiprocessing

import numpy

from ..sources import DerivedSource, VanillaSource, ChunkSource
from . import Supplier
from ..utils import package
from ..utils import count_lines
from ..utils import get_audio_features
from ..utils import Normalize

logger = logging.getLogger(__name__)

###############################################################################
def _init_data_worker():
	import signal
	signal.signal(signal.SIGINT, signal.SIG_IGN)

###############################################################################
def _load_single(args):
	"""
	This function is called through an instance of multiprocessing.Pool 
	in the RawUtterance source.  We do not make this function an instance
	method of RawUtterance in order to avoid unnecessary pickling of 
	RawUtterance instances which would fail due to the presence of some
	unpickleable instance variables.
	"""
	(feature_type, high_freq, on_error), paths = args
	result = [None] * len(paths)
	for i, path in enumerate(paths):
		result[i] = get_audio_features(
			path,
			feature_type=feature_type,
			high_freq=high_freq,
			on_error=on_error
		)
		if result[i] is None:
			logger.error('Failed to load audio file at path: %s', path)
	return result


###############################################################################
def loop_copy(src, dest):
	""" Copies a source array into a destination array, looping
		through the source array until the entire destination array
		is written.

		# Arguments

		src: numpy array. The source array.
		dest: numpy array. The destination array.

		# Return value

		The destination array.

		# Notes

		If the source array is longer than the destination array, then
		only the first N entries of the source array are copied into the
		destination, where N is the length of the destination array.

		If the source array is shorter than, or of equal length with, the
		destination array, then it will be copied into the destination array
		and repeated, back-to-back, until the destination is filled up (almost
		tiling the destination, if you will).
	"""
	if len(src) > len(dest):
		dest[:] = src[:len(dest)]
	else:
		i = 0
		while i < len(dest):
			entries_to_copy = min(len(dest) - i, len(src))
			dest[i:i+entries_to_copy] = src[:entries_to_copy]
			i += len(src)

	return dest

###############################################################################
class UtteranceLength(DerivedSource):
	""" Data source for audio lengths.
	"""
	def __init__(self, source):
		super().__init__()
		self.source = source
	def derive(self, inputs):
		utterances, = inputs
		return numpy.array([[len(x)] for x in utterances], dtype='int32')
	def shape(self):
		return (1, )
	def requires(self):
		return (self.source, )

###############################################################################
class Utterance(DerivedSource):
	""" Data source for model-ready audio samples. Unlike `RawUtterance`, this
		ensures that all data products are rectangular tensors (rather than
		ragged arrays).
	"""
	def __init__(self, source, raw, fill=None, bucket=None):
		super().__init__()
		self.source = source
		self.raw = raw

		self.fill = fill or 'zero'
		assert isinstance(self.fill, str) and self.fill in ('zero', 'loop')
		logger.debug('Utterance source is using fill mode "%s".', self.fill)

		if bucket:
			if not isinstance(bucket, (float, int)) or bucket <= 0:
				raise ValueError('"bucket" must be a positive number.')
			logger.debug('Utterance source is using bucket: %.3f sec', bucket)
			# 10ms audio frames mean there are 100 frames per second.
			# So convert seconds to frames.
			bucket = round(bucket * 100)
			logger.trace('This bucket is equivalent to: %d frames', bucket)
		self.bucket = bucket

	def derive(self, inputs):
		utterances, = inputs
		max_len = max(len(x) for x in utterances)
		if self.bucket:
			max_len = (max_len-1) - ((max_len-1) % self.bucket) + self.bucket

		if self.fill == 'zero':
			output = numpy.zeros(
				shape=(len(utterances), max_len, self.raw.features)
			)
			for i, row in enumerate(utterances):
				output[i][:len(row)] = row
		elif self.fill == 'loop':
			output = numpy.empty(
				shape=(len(utterances), max_len, self.raw.features)
			)
			for i, row in enumerate(utterances):
				loop_copy(row, output[i])
		else:
			raise ValueError('Unhandled fill type: "{}". This is a bug.'
				.format(self.fill))

		return output
	def shape(self):
		return (None, self.raw.features)
	def requires(self):
		return (self.source, )

###############################################################################
class RawUtterance(ChunkSource):
	""" Data source for audio samples
	"""

	DEFAULT_NORMALIZATION_DEPTH = 100
	_pool = None
	_data_cpus = None

	###########################################################################
	@property
	def pool(self):
		return RawUtterance._pool

	###########################################################################
	@property
	def data_cpus(self):
		return RawUtterance._data_cpus

	###########################################################################
	@classmethod
	def _init_pool(cls, data_cpus):
		assert isinstance(data_cpus, int)
		data_cpus = max(1, data_cpus)
		if cls._pool is None:
			cls._data_cpus = data_cpus
			cls._pool = multiprocessing.Pool(data_cpus, _init_data_worker)
		else:
			if data_cpus != cls._data_cpus:
				logger.warning('"data_cpus" has already been set to %d.',
					cls._data_cpus)

	###########################################################################
	@classmethod
	def default_chunk_size(cls):
		""" Returns the default chunk size for this source.
		"""
		return ChunkSource.USE_BATCH_SIZE

	###########################################################################
	def __init__(self, audio_paths, feature_type=None,
		normalization=None, max_frequency=None, data_cpus=1, *args, **kwargs):
		""" Creates a new raw utterance source.
		"""

		super().__init__(*args, **kwargs)
		self.audio_paths = audio_paths
		self.indices = numpy.arange(len(self))
		self.feature_type = feature_type
		self.features = None
		self.max_frequency = max_frequency

		self._init_pool(data_cpus)

		self._init_normalizer(normalization)

	###########################################################################
	def _init_normalizer(self, params):

		# Parse the specification.
		if isinstance(params, str):
			params = {'path' : params}
		elif params is None:
			params = {'path' : None}
		elif not isinstance(params, dict):
			raise ValueError('Unknown normalization value: {}'.format(params))

		# Merge in the defaults.
		defaults = {
			'path' : None,
			'center' : True,
			'scale' : True,
			'rotate' : True,
			'depth' : RawUtterance.DEFAULT_NORMALIZATION_DEPTH
		}
		defaults.update(params)
		params = defaults

		# Create the normalizer.
		norm = Normalize(
			center=params['center'],
			scale=params['scale'],
			rotate=params['rotate']
		)

		path = params['path']
		if path is None:
			logger.warning('No normalization data available. We will use '
				'on-the-fly (non-persistent) normalization. In the future, '
				'you probably want to give a filename to the "normalization" '
				'key in the speech recognition supplier.')
			self.train_normalizer(norm, depth=params['depth'])
		else:
			path = os.path.expanduser(os.path.expandvars(path))
			if os.path.exists(path):
				if not os.path.isfile(path):
					raise ValueError('Normalization data must be a regular '
						'file. This is not: {}'.format(path))
				logger.debug('Restoring normalization statistics: %s', path)
				norm.restore(path)
				self.features = norm.get_dimensionality()
			else:
				logger.info('Training new normalization statistics: %s', path)
				self.train_normalizer(norm, depth=params['depth'])
				norm.save(path)

		# Register the normalizer
		self.norm = norm

	###########################################################################
	def load_audio(self, partial_paths):
		""" Loads unnormalized audio data.
		"""

		# Resolve each path.
		paths = [
			SpeechRecognitionSupplier.find_audio_path(partial_path)
			for partial_path in partial_paths
		]

		for path, partial_path in zip(paths, partial_paths):
			if path is None:
				logger.error('Could not find audio file that---ignoring '
						'extension---begins with: %s', partial_path)
		paths = [path for path in paths if path]

		n_paths = len(paths)
		n_cpus = min(n_paths, self.data_cpus)
		n = n_paths // n_cpus  # paths per cpu
		args = (self.feature_type, self.max_frequency, 'suppress')  # arguments to be passed into worker function

		# Split the paths to be processed as evenly as possible 
		x = [(args, paths[i * n:(i+1) * n]) for i in range(n_cpus - 1)]

		# In case n_paths is not evenly divisible by n_cpus, we handle the last
		# element outside of the above list comprehension
		x.append((args, paths[(n_cpus - 1) * n:])) 
	
		# actually load audio via process pool
		try:
			result = self.pool.map(_load_single, x)
		except KeyboardInterrupt:
			self.pool.terminate()
			self.pool.join()

		# flatten the result, which is a list of lists
		result  =  [x for y in result for x in y]

		# Clean up bad audio
		if any(x is None for x in result):
			logger.warning('Recovering from a bad audio uttereance.')
			good = None
			for candidate in result:
				if candidate is not None:
					good = candidate
					break
			if good is None:
				raise ValueError(
					'Cannot tolerate an entire batch of bad audio.')
			for i, x in enumerate(result):
				if x is None:
					result[i] = good

		return result

	###########################################################################
	def train_normalizer(self, norm, depth):
		""" Trains the normalizer on the data.

			# Arguments

			norm: Normalize instance. The normalization transform to train.
		"""
		logger.debug('Training normalization transform.')
		num_entries = min(depth, len(self.audio_paths))
		paths = random.sample(self.audio_paths, num_entries)
		data = self.load_audio(paths)
		self.features = data[0].shape[-1]
		norm.learn(data)
		logger.debug('Finished training normalization transform.')

	###########################################################################
	def __iter__(self):
		""" Return an iterator to the data.
		"""
		start = 0
		num_entries = len(self)
		while start < num_entries:
			end = min(num_entries, start + self.chunk_size)

			paths = [self.audio_paths[i] for i in self.indices[start:end]]
			batch = self.load_audio(paths)
			batch = [self.norm.apply(data) for data in batch]
			yield batch
			start = end

	###########################################################################
	def __len__(self):
		""" Returns the total number of entries that this source can return, if
			known.
		"""
		return len(self.audio_paths)

	###########################################################################
	def shape(self):
		""" Return the shape of the tensor (excluding batch size) returned by
			this data source.
		"""
		return (None, self.features)

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

###############################################################################
class TranscriptLength(DerivedSource):
	""" Data source for computing transcript lengths.
	"""
	def __init__(self, source):
		super().__init__()
		self.source = source
	def derive(self, inputs):
		transcript, = inputs
		return numpy.array([[len(x)] for x in transcript], dtype='int32')
	def shape(self):
		return (1, )
	def requires(self):
		return (self.source, )

###############################################################################
class Transcript(DerivedSource):
	""" Data source for neat (non-ragged) transcript arrays.
	"""
	def __init__(self, source):
		super().__init__()
		self.source = source
	def derive(self, inputs):
		transcript, = inputs
		max_len = max(len(x) for x in transcript)
		output = numpy.zeros(shape=(len(transcript), max_len), dtype='int32')
		for i, row in enumerate(transcript):
			output[i][:len(row)] = row
		return output
	def shape(self):
		return (None, )
	def requires(self):
		return (self.source, )

###############################################################################
class RawTranscript(ChunkSource):
	""" Data source for variable-length transcripts.
	"""

	###########################################################################
	def __init__(self, transcripts, vocab=None, *args,
		**kwargs):
		""" Creates a new raw transcript source.
		"""
		super().__init__(*args, **kwargs)
		self.transcripts = transcripts
		self.indices = numpy.arange(len(self))
		self.vocab = self.make_vocab(vocab)

	###########################################################################
	@staticmethod
	def make_lower(entry):
		""" Maps strings or lists of strings to lowercase.
		"""
		if isinstance(entry, str):
			return entry.lower()
		return [x.lower() for x in entry]

	###########################################################################
	def make_vocab(self, vocab):
		""" Loads or infers a vocabulary.
		"""
		if vocab is None:
			logger.warning('Inferring vocabulary from data set.')
			data = set(x for transcript in self.transcripts \
				for x in self.make_lower(transcript))
			data = sorted(data)

		elif isinstance(vocab, str):
			logger.debug('Load vocabulary from a JSON file: %s', vocab)
			with open(vocab) as fh:
				json_data = json.loads(fh.read())
			try:
				data = [self.make_lower(x) for x in json_data]
			except:
				logger.exception('Expected the JSON to contain a single list '
					'of strings. Instead, we got: %s', json_data)
				raise

		elif isinstance(vocab, (tuple, list)):
			logger.debug('Using a hard-coded vocabulary.')
			try:
				data = [self.make_lower(x) for x in vocab]
			except:
				logger.exception('Expected the vocabulary to be a list of '
					'strings. Instead, we got: %s', vocab)
				raise

		else:
			raise ValueError('Unknown vocabulary format: {}'.format(vocab))

		if len(set(data)) != len(data):
			raise ValueError('The vocabulary must contain unique entries, but '
				'we found duplicates. Make sure that all entries are unique, '
				'ignoring capitalization. That means you should not have both '
				'"x" and "X" in your vocabulary. For reference, this is the '
				'vocabulary we ended up with: {}'.format(data))

		logger.debug('Loaded a %d-word vocabulary.', len(data))
		return {x : i for i, x in enumerate(data)}

	###########################################################################
	def word_to_integer(self, data):
		""" Maps a character transcript to its integer representation.
		"""
		result = [None]*len(data)
		for i, row in enumerate(data):
			result[i] = [self.vocab.get(word.lower()) for word in row]
			result[i] = [x for x in result[i] if x is not None]
		return result

	###########################################################################
	def __iter__(self):
		""" Return an iterator to the data.
		"""
		start = 0
		num_entries = len(self)
		while start < num_entries:
			end = min(num_entries, start + self.chunk_size)

			batch = [self.transcripts[i] for i in self.indices[start:end]]
			batch = self.word_to_integer(batch)
			yield batch
			start = end

	###########################################################################
	def __len__(self):
		""" Returns the total number of entries that this source can return, if
			known.
		"""
		return len(self.transcripts)

	###########################################################################
	def shape(self):
		""" Return the shape of the tensor (excluding batch size) returned by
			this data source.
		"""
		return (None, )

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

###############################################################################
class SpeechRecognitionSupplier(Supplier):
	""" A supplier which handles parsing of audio + transcript data sets for
		speech recognition purposes.
	"""

	DEFAULT_UNPACK = True
	DEFAULT_TYPE = 'spec'
	SUPPORTED_TYPES = ('wav', 'mp3', 'flac')

	###########################################################################
	@staticmethod
	def find_audio_path(partial_path):
		""" Resolves the audio file extension.
		"""
		for ext in SpeechRecognitionSupplier.SUPPORTED_TYPES:
			candidate = '{}.{}'.format(partial_path, ext)
			if os.path.isfile(candidate):
				return candidate
		return None

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the supplier.
		"""
		return 'speech_recognition'

	###########################################################################
	def __init__(self, url=None, path=None, checksum=None, unpack=None, 
		type=None, normalization=None, min_duration=None, max_duration=None,
		max_frequency=None, vocab=None, samples=None, fill=None, key=None,
		bucket=None, data_cpus=None, *args, **kwargs):
		""" Creates a new speech recognition supplier.

			# Arguments
		"""
		super().__init__(*args, **kwargs)

		if unpack is None:
			unpack = SpeechRecognitionSupplier.DEFAULT_UNPACK
		self.load_data(url=url, path=path, checksum=checksum, unpack=unpack,
			min_duration=min_duration, max_duration=max_duration, text_key=key)
		self.downselect(samples)

		logger.trace('Creating sources.')
		data_cpus = max(1, multiprocessing.cpu_count() - 1 if not data_cpus else data_cpus)
		utterance_raw = RawUtterance(
			self.data['audio'],
			feature_type=type or SpeechRecognitionSupplier.DEFAULT_TYPE,
			normalization=normalization,
			max_frequency=max_frequency,
			data_cpus=data_cpus
		)
		self.sources = {
			'utterance_raw' : utterance_raw,
			'utterance_length' : UtteranceLength('utterance_raw'),
			'utterance' : Utterance(
				'utterance_raw',
				utterance_raw,
				fill=fill,
				bucket=bucket
			),
			'duration' : VanillaSource(numpy.array(self.data['duration'])),
			'audio_source' : VanillaSource(numpy.array(self.data['audio']))
		}
		for i, (k, v) in enumerate(self.data['transcript'].items()):
			logger.trace('Parsing the vocabulary for key: %s', k)
			if len(self.data['transcript']) == 1:
				prefix = ''
			else:
				prefix = '{}_'.format(k)

			if isinstance(vocab, dict):
				if k in vocab:
					this_vocab = vocab[k]
				else:
					raise ValueError('If the vocabulary is a dictionary, then '
						'it must have keys corresponding to the text keys.')
			elif isinstance(vocab, (list, tuple)):
				if all(isinstance(v, (list, tuple)) for v in vocab):
					this_vocab = vocab[i]
				else:
					this_vocab = vocab
			else:
				this_vocab = vocab

			raw_name = '{}transcript_raw'.format(prefix)
			self.sources.update({
				raw_name : RawTranscript(
					v,
					vocab=this_vocab
				),
				'{}transcript_length'.format(prefix) : TranscriptLength(
					raw_name
				),
				'{}transcript'.format(prefix) : Transcript(raw_name)
			})

	###########################################################################
	def downselect(self, samples):
		""" Selects a subset of the data.

			# Arguments

			samples: None, int, or str. If None, uses all samples. If an
				integer, uses the first `samples` entries. If a string, follows
				the `Sample Specification` below.

			# Sample Specification

			Forms:			Meaning:

			10				Use 10 samples: 0 through 10.
			10-				Use all but 10 samples: use the 10th through
							the remainder.
			10%				Use the first 10% of samples.
			10-20			Use 10 samples: 10 through 19.
			10-20%			Use 10% of samples, from 10% through 20%.
							When combined with a random seed, this lets you
							split a dataset on the fly.
			10-%			Use all but 10% of samples: from 10% through
							the remainder.

			# Notes:

			- For percentage ranges of the form "X-%" or "X-Y%", Kur will
			  compute the percentage and then add one to the start value.
			  This makes it easier for you to use a random seed and then
			  make dataset splits like 10%, 10-20%, 20% without worrying
			  about having disjoint datasets.

			- As evidenced by the examples above, Kur follows Python in using
			  ranges that exclude the upper bound (e.g., "10" means "0 .. 9"
			  and "10-20" means "10..19").
		"""
		if samples is None:
			logger.trace('Using all available data.')
			return
		elif isinstance(samples, int):
			if samples < 1:
				raise ValueError('"samples" cannot be less than 1.')
			if samples >= self.metadata['entries']:
				return
			logger.debug('Using only %d / %d samples of the available data.',
				samples, self.metadata['entries'])
			start = 0
			end = samples
		elif isinstance(samples, str):
			regex = re.compile(
				r'(?P<start>[0-9]+(?:\.[0-9]*)?)'
				r'(?:(?P<range>-)'
					r'(?P<end>[0-9]+(?:\.[0-9]*)?)?'
				r')?'
				r'(?P<unit>%)?'
			)
			match = regex.match(samples)
			if not match:
				raise ValueError('Failed to parse the "samples" '
					'specification: {}'.format(samples))

			result = match.groupdict()
			start = float(result['start'])
			if result['range']:
				if result['end']:
					end = float(result['end'])
				elif result['unit']:
					end = 100
				else:
					end = self.metadata['entries']
			else:
				end = start
				start = 0

			if result['unit']:
				start = int(self.metadata['entries'] * (start / 100))
				end = int(self.metadata['entries'] * (end / 100))
			else:
				start = int(start)
				end = int(end)

			start = min(max(0, start), self.metadata['entries'])
			end = min(max(0, end), self.metadata['entries'])

			if start == 0 and end == self.metadata['entries']:
				return

			if start >= end:
				raise ValueError('No samples pass this "samples" cut: [{}, {})'
					.format(start, end))
		else:
			raise TypeError('Invalid/unexpected type for "samples": {}'
				.format(samples))

		# Create the seeded random number generator.
		gen = numpy.random.RandomState(
			seed=self.kurfile.get_seed() if self.kurfile else None
		)

		# Produce a mask (True = keep, False = discard)
		mask = numpy.zeros(self.metadata['entries'], dtype=bool)
		indices = gen.permutation(self.metadata['entries'])[start:end]
		mask[indices] = True

		# Downselect
		for k in self.data:
			if isinstance(self.data[k], dict):
				for inner in self.data[k]:
					self.data[k][inner] = [
						x for i, x in enumerate(self.data[k][inner]) if mask[i]
					]
			else:
				self.data[k] = [
					x for i, x in enumerate(self.data[k]) if mask[i]
				]

		self.metadata['entries'] = int(end - start)

	###########################################################################
	def load_data(self, url=None, path=None, checksum=None, unpack=None,
		min_duration=None, max_duration=None, text_key=None):
		""" Loads the data for this supplier.
		"""
		logger.info('Loading input dataset: %s', path if path else url)
		local_path, is_packed = package.install(
			url=url,
			path=path,
			checksum=checksum
		)

		manifest = None
		if is_packed and unpack:
			logger.trace('Unpacking input data: %s', local_path)
			manifest = package.unpack(local_path, recursive=True)
			is_packed = False
		elif is_packed and not unpack:
			logger.trace('Using packed input data.')
			raise NotImplementedError
		elif not is_packed and unpack:
			logger.trace('Using already unpacked input data.')
		elif not is_packed and not unpack:
			logger.trace('Ignore "unpack" for input data, since it is already '
				'unpacked.')
		else:
			logger.error('Unhandled data package requirements. This is a bug.')

		self.metadata, self.data = self.get_metadata(
			manifest=manifest,
			root=local_path,
			min_duration=min_duration,
			max_duration=max_duration,
			text_key=text_key
		)

	###########################################################################
	def get_metadata(self, manifest=None, root=None, min_duration=None,
		max_duration=None, text_key=None):
		""" Scans the package for a metadata file, makes sure everything is in
			order, and returns some information about the data set.
		"""
		logger.trace('Looking for metadata file.')
		metadata_file = None

		if not text_key:
			text_key = ('text', )
		elif isinstance(text_key, str):
			text_key = (text_key, )
		elif isinstance(text_key, list):
			text_key = tuple(text_key)
		elif not isinstance(text_key, tuple):
			raise ValueError('"key" must be a string, None, or a list. '
				'Instead, we received: {}'.format(text_key))

		def look_in_list(filenames):
			""" Searches a list of files for a JSONL file.
			"""
			for filename in filenames:
				parts = os.path.splitext(filename)
				if parts[1].lower() == '.jsonl' and \
					not os.path.basename(filename).startswith('.'):
					return filename
			return None

		if manifest is None:
			if root is None:
				raise ValueError('No root provided and no manifest provided. '
					'This is a bug.')
			if not os.path.isdir(root):
				raise ValueError('Root is not a directory. This is a bug.')
			for dirpath, _, filenames in os.walk(root):
				metadata_file = look_in_list(filenames)
				if metadata_file is not None:
					metadata_file = os.path.join(dirpath, metadata_file)
					break
		else:
			metadata_file = look_in_list(manifest)

		if metadata_file is None:
			raise ValueError('Failed to find a JSONL metadata file.')

		source = os.path.join(
			os.path.dirname(metadata_file),
			'audio'
		)

		logger.debug('Found metadata file: %s', metadata_file)
		logger.debug('Inferred source path: %s', source)

		logger.debug('Scanning metadata file.')
		lines = count_lines(metadata_file)
		logger.debug('Entries counted: %d', lines)

		logger.debug('Loading metadata.')
		data = {
			'audio' : [None]*lines,
			'transcript' : {k : [None]*lines for k in text_key},
			'duration' : [None]*lines
		}

		entries = 0
		valid_entries = 0
		required_keys = text_key + ('duration_s', 'uuid')
		with open(metadata_file, 'r') as fh:
			for line_number, line in enumerate(fh, 1):
				if line_number >= 10 and valid_entries < line_number/2:
					raise ValueError('Data file has too many bad entries: {}'
						.format(metadata_file))

				try:
					entry = json.loads(line)
				except json.decoder.JSONDecodeError:
					logger.warning('Failed to parse valid JSON on line %d of '
						'file %s', line_number, metadata_file)
					continue

				bad = False
				for k in required_keys:
					if k not in entry:
						logger.warning('Line %d is missing one of its '
							'required keys in metadata file %s: %s. Available '
							'keys are: %s', line_number, metadata_file, k,
							', '.join(entry.keys()))
						bad = True
						break
				if bad:
					continue

				valid_entries += 1

				duration = entry['duration_s']
				if min_duration and duration < min_duration:
					continue
				if max_duration and duration > max_duration:
					continue

				data['duration'][entries] = entry['duration_s']
				for k in text_key:
					data['transcript'][k][entries] = entry[k]
				data['audio'][entries] = os.path.join(source, entry['uuid'])

				entries += 1

		logger.debug('Entries kept: %d', entries)
		for k in data:
			if isinstance(data[k], dict):
				for inner in data[k]:
					data[k][inner] = data[k][inner][:entries]
			else:
				data[k] = data[k][:entries]

		metadata = {
			'entries' : entries,
			'filename' : metadata_file,
			'source' : source
		}

		return metadata, data

	###########################################################################
	def get_sources(self, sources=None):
		""" Returns all sources from this provider.
		"""

		if sources is None:
			sources = list(self.sources.keys())
		elif not isinstance(sources, (list, tuple)):
			sources = [sources]

		for source in sources:
			if source not in self.sources:
				raise KeyError(
					'Invalid data key: {}. Valid keys are: {}'.format(
						source, ', '.join(str(k) for k in self.sources.keys())
				))

		return {k : self.sources[k] for k in sources}

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

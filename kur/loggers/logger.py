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
from collections import deque
import numpy
from ..utils import get_subclasses, Timer, CriticalSection

logger = logging.getLogger(__name__)

###############################################################################
class Logger:
	""" Base class for loggers.

		Loggers are special callbacks intended to process training statistics
		(such as loss).
	"""

	###########################################################################
	@staticmethod
	def from_specification(spec):
		""" Creates a new evaluation hook from a specification.

			# Arguments

			spec: str or dict. The specification object that was given.

			# Return value

			New EvaluationHook instance.
		"""
		if isinstance(spec, str):
			cls = Logger.DEFAULT_LOGGER
			params = {'path' : spec}
		elif isinstance(spec, dict):
			spec = dict(spec)
			name = spec.pop('name', None)
			if name is None:
				cls = Logger.DEFAULT_LOGGER
			else:
				cls = Logger.get_logger_by_name(name)
			params = spec
		elif spec is None:
			return None
		else:
			raise ValueError('Expected the logger to be a string or '
				'dictionary. We got this instead: {}'.format(spec))

		return cls(**params)

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the logger.

			# Return value

			A lower-case string unique to this logger.
		"""
		return cls.__name__.lower()

	###########################################################################
	@staticmethod
	def get_all_loggers():
		""" Returns an iterator to the names of all logger.
		"""
		for cls in get_subclasses(Logger):
			yield cls

	###########################################################################
	@staticmethod
	def get_logger_by_name(name):
		""" Finds a logger class with the given name.
		"""
		name = name.lower()
		for cls in Logger.get_all_loggers():
			if cls.get_name() == name:
				return cls
		raise ValueError('No such logger with name "{}"'.format(name))

	###########################################################################
	def __init__(self, keep_batch=True, rate=None, **kwargs):
		""" Creates a new logger.

			# Arguments

			keep_batch: bool (default: True). If True, batch-level statistics
				are kept. Otherwise, they are silently discarded. Setting this
				to False can help with logging performance.
			rate: int or None (default: None). If `keep_batch` is True, then
				batch-level statistics will not be written more often than
				once every `rate` seconds (unless an epoch is completed, in
				which case batch statistics are written regardless of the value
				of `rate`, assuming `keep_batch` is True). If `rate` is zero,
				then batch information is written as quickly as possible (every
				batch). If `rate` is None, then, batch information is written
				only every epoch (equivalent to an arbitrarily large value of
				`rate`).
		"""
		if kwargs:
			logger.warning('Unexpected or unsupported arguments to the '
				'logger: %s', ', '.join(str(k) for k in kwargs))

		self.keep_batch = keep_batch
		if rate is not None:
			if not keep_batch:
				logger.warning('Logger rate is only meaningful when '
					'keep_batch is True. Ignoring the rate value.')
				rate = None
			if not isinstance(rate, int):
				raise ValueError('Logger rate must be None or an integer. '
					'Instead, we got: {}'.format(rate))
		self.rate = rate

		self.data = None
		self.batches = 0
		self.epochs = 0
		self.samples = 0
		self.timer = Timer(started=False)
		self.timestamper = Timer(started=False)

		self.latest_batch_loss = None
		self.latest_epoch_loss = None
		self.latest_training_loss = None
		self.latest_validation_loss = None
		self.new_epoch = True
		self.samples_this_epoch = 0

		self.clocks = None

		self._clear()

	###########################################################################
	def _clear(self):
		""" Clears the stored data.
		"""
		self.data = {k : {} for k in ('batch', 'training', 'validation')}

	###########################################################################
	@staticmethod
	def _get_dtype(entry):
		""" Finds an appropriate numpy datatype for the given data value.
		"""
		if isinstance(entry, numpy.ndarray):
			return entry.dtype
		elif isinstance(entry, (list, tuple)):
			if entry:
				return Logger._get_dtype(entry[0])
		elif isinstance(entry, int):
			return numpy.dtype('int32')

		# Floats, empty lists, and "other" types all get mapped to this default.
		return numpy.dtype('float32')

	###########################################################################
	@staticmethod
	def _get_shape(entry):
		""" Determines the shape of a data value.
		"""
		if isinstance(entry, numpy.ndarray):
			return entry.shape
		elif isinstance(entry, (list, tuple)):
			if entry:
				return (len(entry), ) + Logger._get_shape(entry[0])

		return ()

	###########################################################################
	@staticmethod
	def _prepare_like(entry, num_entries):
		""" Allocates a numpy array to hold a certain number of data values.
		"""
		return numpy.empty(
			shape=(num_entries, ) + Logger._get_shape(entry),
			dtype=Logger._get_dtype(entry)
		)

	###########################################################################
	@staticmethod
	def _arrange(data):
		""" Neatly arranges data in a Numpy array.

			# Arguments

			data: deque. Each entry in the deque is a dictionary whose keys name
				the training statistic stored in the respective dictionary
				values.

			# Return value

			Dictionary with the same keys as the entries in `data`, but whose
			values are each an array containing the concatenated values from
			the respective keys in `data`.
		"""
		num_entries = len(data)
		result = None
		for i, entry in enumerate(data):
			if result is None:
				result = {k : Logger._prepare_like(v, num_entries)
					for k, v in entry.items()}
			for k, v in entry.items():
				result[k][i] = v
		return result

	###########################################################################
	def flush(self):
		""" Hook for asking the logger to process log information in its queue.
		"""
		with CriticalSection():
			for key, tags in self.data.items():
				for tag, data in tags.items():
					if not data:
						continue
					data = self._arrange(data)
					self.process(data, key, tag)

			self._clear()
			self.timer.restart()

	###########################################################################
	def _push(self, data_type, tag, data):
		""" Helper function for pushing data into the storage queue.
		"""
		data = dict(data)
		if not isinstance(tag, tuple):
			tag = (tag, )
		if tag not in self.data[data_type]:
			self.data[data_type][tag] = deque()
		if tag[0].startswith('loss'):
			if 'total' not in data:
				data['total'] = sum(data.values())
		data['batch'] = self.batches
		data['time'] = self.timestamper()
		self.timestamper.resume()
		self.data[data_type][tag].append(data)

	###########################################################################
	def record_clocks(self, clocks):
		""" Take a snapshot of any timer values that should be logged.
		"""
		self.clocks = clocks

	###########################################################################
	def get_latest_training_loss(self, reduced=True):
		if self.latest_training_loss is None:
			return None
		if reduced:
			return sum(self.latest_training_loss.values())
		return self.latest_training_loss

	###########################################################################
	def get_latest_batch_loss(self):
		return self.latest_batch_loss

	###########################################################################
	def get_latest_validation_loss(self):
		return self.latest_validation_loss

	###########################################################################
	def get_latest_epoch_loss(self):
		return self.latest_epoch_loss

	###########################################################################
	def get_samples_this_epoch(self):
		return self.samples_this_epoch

	###########################################################################
	def log_batch(self, batch_size, data, tag=None, *, clocks=None):
		""" Log training information after a batch.
		"""
		self.latest_batch_loss = sum(data.values())

		if self.new_epoch:
			self.new_epoch = False
			self.samples_this_epoch = batch_size
			self.latest_training_loss = data
		else:
			new_entries = self.samples_this_epoch + batch_size
			self.latest_training_loss = {
				k : v * (self.samples_this_epoch / new_entries) + \
					data[k] * (batch_size / new_entries)
				for k, v in self.latest_training_loss.items()
			}
			self.samples_this_epoch = new_entries

		if clocks is not None:
			self.record_clocks(clocks)
		self.samples += batch_size
		self.batches += 1
		if not self.keep_batch:
			return

		self._push('batch', tag, data)

		if not self.timer.started or \
				(self.rate is not None and self.timer.get() > self.rate):
			self.flush()

	###########################################################################
	def log_training(self, data, tag=None, *, clocks=None):
		""" Log training statistics after an epoch.
		"""
		self.latest_epoch_loss = sum(data.values())
		self.new_epoch = True

		if clocks is not None:
			self.record_clocks(clocks)
		self.epochs += 1
		self._push('training', tag, data)
		self.flush()

	###########################################################################
	def log_validation(self, data, tag=None, *, clocks=None):
		""" Log training statistics after a validation run.
		"""
		self.latest_validation_loss = sum(data[None].values())

		if clocks is not None:
			self.record_clocks(clocks)
		for k, v in data.items():
			self._push('validation', (tag, k) if k else tag, v)
		self.flush()

	###########################################################################
	def get_number_of_epochs(self):
		""" Returns the number of epochs this model has historically completed.

			# Return value

			If the number of epochs is known from the logs, it is returned.
			Otherwise, the count is started from zero.
		"""
		return self.epochs

	###########################################################################
	def get_number_of_batches(self):
		""" Returns the number of batches this model has historically
			completed.

			# Return value

			If the number of epochs is known from the logs, it is returned.
			Otherwise, the count is started from zero.
		"""
		return self.batches

	###########################################################################
	def get_number_of_samples(self):
		""" Returns the number of samples this model has historically
			trained on.

			# Return value

			If the number of epochs is known from the logs, it is returned.
			Otherwise, the count is started from zero.
		"""
		return self.samples

	###########################################################################
	def process(self, data, data_type, tag=None):
		""" Processes training statistics.

			This should be implemented in derived classes, and provides the
			primary point for derived classes to receive data. This layer of
			indirection between the `log_X()` functions and this `process()`
			function is an optimization that improves performance by throttling
			the rate at which log data is processed.

			# Arguments

			data: dict. The training statistics dictionary, whose keys name the
				particular statistic, and the values are numpy arrays containing
				the most recent log data.
			data_type: str (one of: "batch", "training", "validation").
				Indicates what stage of training this statistics represent.
			tag: str or None (default: None). A tag to describe the type of data
				provided by `data` (e.g., "loss" or "accuracy").

			# Return value

			None

			# Notes

			- Although the `log_X()` functions accept dictionary values which
			  may be raw floats, `process()` only ever deals with values that
			  are arrays. The base class will collate the information into these
			  arrays so that they can be efficiently parsed by the child
			  classes.
		"""
		raise NotImplementedError

# Need to put this important here to avoid circular dependencies.
from .binary_logger import BinaryLogger	# pylint: disable=wrong-import-position
Logger.DEFAULT_LOGGER = BinaryLogger

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

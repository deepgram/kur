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
import re
import shutil
import logging

import yaml

from .persistent_logger import PersistentLogger
from .statistic import Statistic
from ..utils import idx

logger = logging.getLogger(__name__)

###############################################################################
class BinaryLogger(PersistentLogger):
	""" A class for storing log data in a fast binary format that can be
		quickly appended to or randomly seeked.
	"""

	SUMMARY = 'summary.yml'

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the logger.
		"""
		return 'binary'

	###########################################################################
	def __init__(self, path, **kwargs):
		""" Creates a new binary logger.

			# Arguments

			path: str. The path to create the log at.

			# Notes

			- The BinaryLogger stores its log information in a directory. Thus,
			  `path` should be a directory.
		"""
		super().__init__(**kwargs)

		self.sessions = 1

		self.path = path
		self.prepare()

	###########################################################################
	def prepare(self):
		""" Prepares the logger by loading historical information.
		"""
		path = os.path.expanduser(os.path.expandvars(self.path))

		if os.path.exists(path):
			if os.path.isdir(path):
				logger.info('Loading log data: %s', path)

				summary_path = os.path.join(path, self.SUMMARY)
				if os.path.exists(summary_path):
					self.load_summary()
					has_summary = True
				else:
					logger.debug('Loading old-style binary logger.')
					has_summary = False

				_, _, training_loss = self.load_statistic(
					Statistic(Statistic.Type.TRAINING, 'loss', 'total')
				)
				if training_loss is None:
					self.best_training_loss = None
				else:
					self.best_training_loss = training_loss.min()

					# Handle the old log format.
					if not has_summary:
						self.epochs = len(training_loss)

				_, _, validation_loss = self.load_statistic(
					Statistic(Statistic.Type.VALIDATION, 'loss', 'total')
				)
				if validation_loss is None:
					self.best_validation_loss = None
				else:
					self.best_validation_loss = validation_loss.min()

			else:
				raise ValueError('Binary logger stores its information in a '
					'directory. The supplied log path already exists, but it '
					'is a file: {}'.format(path))

		else:
			logger.info('Log does not exist. Creating path: %s', path)
			os.makedirs(path, exist_ok=True)

			self.best_training_loss = None
			self.best_validation_loss = None

	###########################################################################
	def get_best_training_loss(self):
		""" Returns the best historical training loss.
		"""
		return self.best_training_loss

	###########################################################################
	def get_best_validation_loss(self):
		""" Returns the best historical validation loss.
		"""
		return self.best_validation_loss

	###########################################################################
	@staticmethod
	def get_filename_from_statistic(statistic, *, legacy=False):
		if legacy:
			return '{}_{}_{}'.format(
				str(statistic.data_type),
				'.'.join(statistic.tags),
				statistic.name
			)
		return '{}_{}_{}'.format(
			str(statistic.data_type).replace('_', '__'),
			'.'.join(
					tag.replace('_', '__').replace('.', '..')
					for tag in statistic.tags
				)
				if statistic.tags else 'null',
			statistic.name.replace('_', '__')
		)

	###########################################################################
	@staticmethod
	def get_statistic_from_filename(filename, *, was_legacy=None):

		parts = re.split(r'(?<=[^_])_(?=[^_])', filename)
		if len(parts) != 3:
			parts = filename.split('_', 1)
			if len(parts) != 2:
				raise KeyError('Failed to parse filename.')
			data_type = parts[0]
			parts = parts[1].rsplit('_', 1)
			if len(parts) != 2:
				raise KeyError('Failed to parse filename.')

			tags, name = parts
			tags = tags.split('.')

			if was_legacy is None:
				logger.warning('Parsed a legacy statistic filename: %s',
					filename)
			else:
				was_legacy.legacy = True
			return Statistic(data_type, tags, name)

		parts = [part.replace('__', '_') for part in parts]

		data_type = parts[0]

		tags = re.split(r'(?<=[^.])\.(?=[^.])', parts[1])
		tags = [tag.replace('..', '.') for tag in tags]
		if len(tags) == 1 and tags[0] == 'null':
			tags = None

		name = parts[2]

		return Statistic(data_type, tags, name)

	###########################################################################
	def process(self, data, data_type, tag=None):
		""" Processes training statistics.
		"""
		path = os.path.expanduser(os.path.expandvars(self.path))
		for k, v in data.items():
			stat = Statistic(data_type, tag, k)

			column = self.get_filename_from_statistic(stat)
			filename = os.path.join(path, column)

			# Handle deprecated paths.
			if not os.path.isfile(filename):
				old_column = self.get_filename_from_statistic(stat,
					legacy=True)
				old_filename = os.path.join(path, old_column)
				if os.path.isfile(old_filename):
					logger.warning('Your log paths use a deprecated naming '
						'scheme. We are upgrading it now, leaving the '
						'original file as a backup: %s', old_filename)
					shutil.copy(old_filename, filename)

			logger.debug('Adding data to binary column: %s', column)
			idx.save(filename, v, append=True)

		self.update_summary()

	###########################################################################
	def update_summary(self):
		""" Updates the summary log file.
		"""
		logger.debug('Writing logger summary.')
		path = os.path.expanduser(os.path.expandvars(self.path))
		summary_path = os.path.join(path, self.SUMMARY)
		with open(summary_path, 'w') as fh:
			fh.write(yaml.dump({
				'version' : 3,
				'epochs' : self.epochs,
				'batches' : self.batches,
				'samples' : self.samples,
				'sessions' : self.sessions,
				'clocks' : {k : float(v) for k, v in self.clocks.items()} \
					if self.clocks else None
			}))

	###########################################################################
	def load_summary(self):
		logger.debug('Reading logger summary.')
		path = os.path.expanduser(os.path.expandvars(self.path))
		summary_path = os.path.join(path, self.SUMMARY)
		with open(summary_path) as fh:
			summary = yaml.load(fh.read())
		self.epochs = summary.get('epochs', 0)
		self.batches = summary.get('batches', 0)
		self.samples = summary.get('samples', 0)
		self.sessions = summary.get('sessions', 0) + 1
		self.clocks = summary.get('clocks')

	###########################################################################
	@staticmethod
	def load_column(path, column):
		""" Loads logged information from disk.

			# Arguments

			path: str. The path to the log directory.
			column: str. The name of the statistic to load.

			# Return value

			If the statistic specified by `column` exists in the log directory
			`path`, then a numpy array containing the stored data is returned.
			Otherwise, None is returned.
		"""
		logger.debug('Loading binary column: %s', column)
		path = os.path.expanduser(os.path.expandvars(path))
		if not os.path.isdir(path):
			logger.debug('No such log path exists: %s', path)
			return None

		filename = os.path.join(path, column)
		if not os.path.isfile(filename):
			logger.debug('No such log column exists: %s', filename)
			return None

		return idx.load(filename)

	###########################################################################
	def enumerate_statistics(self):

		result = []

		legacy = lambda: None

		path = os.path.expanduser(os.path.expandvars(self.path))
		for filename in os.listdir(path):
			if filename == self.SUMMARY:
				continue

			if not os.path.isfile(os.path.join(path, filename)):
				continue

			try:
				legacy.legacy = False
				stat = self.get_statistic_from_filename(filename,
					was_legacy=legacy)
				if legacy.legacy and os.path.isfile(os.path.join(
					path,
					self.get_filename_from_statistic(stat)
				)):
					# If this is a legacy filename, but a "modern" filename
					# exists, then just enumerate the modern one and skip this.
					continue
			except KeyError:
				continue

			result.append(stat)

		return result

	###########################################################################
	def load_statistic(self, statistic):
		path = os.path.expanduser(os.path.expandvars(self.path))

		def load_with_legacy_support(statistic):
			result = BinaryLogger.load_column(path,
				self.get_filename_from_statistic(statistic)
			)
			if result is None:
				result = BinaryLogger.load_column(path,
					self.get_filename_from_statistic(statistic, legacy=True)
				)
				if result is not None:
					logger.warning('Loading deprecated log path.')
			return result

		values = load_with_legacy_support(statistic)
		batches = load_with_legacy_support(statistic.copy(name='batch'))
		timestamps = load_with_legacy_support(statistic.copy(name='time'))

		lens = [len(x) for x in (values, batches, timestamps) if x is not None]
		if not lens:
			return (None, None, None)
		keep = min(lens)

		def modify(x):
			""" Truncates the data series appropriately.
			"""
			if x is None:
				return None
			elif keep:
				return x[-keep:]
			else:
				return x[0:0]

		values, batches, timestamps = \
			tuple(modify(x) for x in (values, batches, timestamps))

		return (batches, timestamps, values)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

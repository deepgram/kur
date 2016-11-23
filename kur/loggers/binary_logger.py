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
import logging
from .logger import Logger
from ..utils import idx

logger = logging.getLogger(__name__)

################################################################################
class BinaryLogger(Logger):
	""" A class for storing log data in a fast binary format that can be
		quickly appended to or randomly seeked.
	"""

	############################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the logger.
		"""
		return 'binary'

	############################################################################
	def __init__(self, path, **kwargs):
		""" Creates a new binary logger.

			# Arguments

			path: str. The path to create the log at.

			# Notes

			- The BinaryLogger stores its log information in a directory. Thus,
			  `path` should be a directory.
		"""
		super().__init__(**kwargs)

		self.path = path
		self.prepare()

	############################################################################
	def prepare(self):
		""" Prepares the logger by loading historical information.
		"""
		path = os.path.expanduser(os.path.expandvars(self.path))

		if os.path.exists(path):
			if os.path.isdir(path):
				logger.info('Loading log data: %s', path)

				training_loss = BinaryLogger.load_column(
					path, 'training_loss_total')
				if training_loss is None:
					self.best_training_loss = None
					self.num_epochs = None
				else:
					self.best_training_loss = training_loss.min()
					self.num_epochs = len(training_loss)

				validation_loss = BinaryLogger.load_column(
					path, 'validation_loss_total')
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
			self.num_epochs = None
			self.best_validation_loss = None

	############################################################################
	def get_best_training_loss(self):
		""" Returns the best historical training loss.
		"""
		return self.best_training_loss

	############################################################################
	def get_best_validation_loss(self):
		""" Returns the best historical validation loss.
		"""
		return self.best_validation_loss

	############################################################################
	def get_number_of_epochs(self):
		""" Returns the number of epochs this model has historically completed.
		"""
		return self.num_epochs

	############################################################################
	def process(self, data, data_type, tag=None):
		""" Processes training statistics.
		"""
		path = os.path.expanduser(os.path.expandvars(self.path))
		for k, v in data.items():
			column = '{}_{}_{}'.format(data_type, tag, k)

			filename = os.path.join(path, column)

			logger.debug('Adding data to binary column: %s', column)
			idx.save(filename, v, append=True)

	############################################################################
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

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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

import os
import logging

import numpy

from . import TrainingHook
from ...loggers import PersistentLogger, Statistic

logger = logging.getLogger(__name__)

###############################################################################
class PlotHook(TrainingHook):
	""" Hook for creating plots of loss.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the hook.
		"""
		return 'plot'

	###########################################################################
	def __init__(self, path, *args, **kwargs):
		""" Creates a new plotting hook.
		"""
		super().__init__(*args, **kwargs)
		self.path = os.path.expanduser(os.path.expandvars(path))

		try:
			import matplotlib					# pylint: disable=import-error
		except:
			logger.exception('Failed to import "matplotlib". Make sure it is '
				'installed, and if you have continued trouble, please check '
				'out our troubleshooting page: https://kur.deepgram.com/'
				'troubleshooting.html#plotting')
			raise

		matplotlib.use('Agg')

	###########################################################################
	def notify(self, status, log=None, info=None):
		""" Creates the plot.
		"""
		from matplotlib import pyplot as plt	# pylint: disable=import-error

		logger.debug('Plotting hook received training message.')

		if status not in (
			TrainingHook.TRAINING_END,
			TrainingHook.VALIDATION_END,
			TrainingHook.EPOCH_END
		):
			logger.debug('Plotting hook does not handle this status.')
			return

		if log is None:
			logger.warning('Plot hook was requested, but no logger is '
				'available.')
			return
		if not isinstance(log, PersistentLogger):
			logger.warning('Plot hook was requested, but the logger does not '
				'track statistics.')
			return

		log.flush()

		plt.xlabel('Batch')
		plt.ylabel('Loss')

		batch, loss = log.load_statistic(
			Statistic(Statistic.Type.BATCH, 'loss', 'total')
		)
		if loss is None:
			batch, loss = log.load_statistic(
				Statistic(Statistic.Type.TRAINING, 'loss', 'total')
			)
			if loss is None:
				logger.debug('No training data available for plotting yet.')
				return
			logger.debug('Using per-epoch training statistics for plotting.')
		else:
			logger.debug('Using per-batch training statistics for plotting.')

		if batch is None:
			batch = numpy.arange(1, len(loss)+1)
		t_line, = plt.plot(batch, loss, 'co-', label='Training Loss')

		batch, loss = log.load_statistic(
			Statistic(Statistic.Type.VALIDATION, 'loss', 'total')
		)
		if batch is None:
			batch = numpy.arange(1, len(loss)+1)
		v_line, = plt.plot(batch, loss, 'mo-', label='Validation Loss')

		plt.legend(handles=[t_line, v_line])
		plt.savefig(self.path, transparent=True, bbox_inches='tight')
		plt.clf()

		logger.debug('Plot saved to: %s', self.path)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

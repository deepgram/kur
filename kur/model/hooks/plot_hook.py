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
	def __init__(self, loss_per_batch=None, loss_per_time=None,
		throughput_per_time=None, *args, **kwargs):
		""" Creates a new plotting hook.
		"""
		super().__init__(*args, **kwargs)

		plots = dict(zip(
			('loss_per_batch', 'loss_per_time', 'throughput_per_time'),
			(loss_per_batch, loss_per_time, throughput_per_time)
		))

		self.plots = plots
		for k, v in self.plots.items():
			if v is not None:
				self.plots[k] = os.path.expanduser(os.path.expandvars(v))

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

		# Load the data
		batch, time, loss = log.load_statistic(
			Statistic(Statistic.Type.BATCH, 'loss', 'total')
		)
		if loss is None:
			batch, time, loss = log.load_statistic(
				Statistic(Statistic.Type.TRAINING, 'loss', 'total')
			)
			if loss is None:
				logger.debug('No training data available for plotting yet.')
				return
			logger.debug('Using per-epoch training statistics for plotting.')
		else:
			logger.debug('Using per-batch training statistics for plotting.')

		vbatch, vtime, vloss = log.load_statistic(
			Statistic(Statistic.Type.VALIDATION, 'loss', 'total')
		)

		path = self.plots.get('loss_per_batch')
		if path:
			plt.xlabel('Batch')
			plt.ylabel('Loss')

			if batch is None:
				batch = numpy.arange(1, len(loss)+1)
			t_line, = plt.plot(batch, loss, 'co-', label='Training Loss')

			if vbatch is None:
				vbatch = numpy.arange(1, len(vloss)+1)
			v_line, = plt.plot(vbatch, vloss, 'mo-', label='Validation Loss')

			plt.legend(handles=[t_line, v_line])
			plt.savefig(path, transparent=True, bbox_inches='tight')
			plt.clf()

			logger.debug('Loss-per-batch plot saved to: %s', path)

		path = self.plots.get('loss_per_time')
		if path:
			plt.xlabel('Time')
			plt.ylabel('Loss')

			if time is None:
				logger.warning('No training timestamps available for '
					'loss-per-time plot.')
				t_line = None
			else:
				t_line, = plt.plot(time, loss, 'co-', label='Training Loss')

			if vtime is None:
				logger.warning('No validation timestamps available for '
					'loss-per-time plot.')
				v_line = None
			else:
				v_line, = plt.plot(vtime, vloss, 'mo-',
					label='Validation Loss')

			plt.legend(handles=[x for x in (t_line, v_line) if x is not None])
			plt.savefig(path, transparent=True, bbox_inches='tight')
			plt.clf()

			logger.debug('Loss-per-time plot saved to: %s', path)

		path = self.plots.get('throughput_per_time')
		if path:
			if time is None or batch is None:
				logger.warning('No training timestamps or batches available '
					'for throughput-per-time plot.')
			else:
				plt.xlabel('Time')
				plt.ylabel('Throughput (Batches/Second)')

				throughput = numpy.diff(batch) / numpy.diff(time)
				t_line, = plt.plot(time[1:], throughput, 'co-',
					label='Training Throughput')

				plt.legend(handles=(t_line, ))
				plt.savefig(path, transparent=True, bbox_inches='tight')
				plt.clf()

				logger.debug('Throughput-per-time plot saved to: %s', path)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

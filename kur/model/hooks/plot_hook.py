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

import colorsys
import os
import logging
import itertools
from collections import OrderedDict

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
	def validation_style(self):
		""" Returns an iterator over the formatting styles used by validation
			plots.
			n is the number of colors to choose from for styling.
		"""
		
		def taste_the_rainbow():
			n_hues = 7
			h = [x * 1.0 / n_hues for x in range(n_hues)]  # All possible hues
			sv = [0.7, 1.0]  # all possible saturation and values
			hsv = [(x, y, y) for x, y in itertools.product(h, sv)]
			return numpy.array(list(map(lambda x: colorsys.hsv_to_rgb(*x) + (1, ), hsv)))
	
		# Too many colors means neighboring colors that are 
		# hard to distinguish between, so we cap n.
		colors = taste_the_rainbow()

		lines = ('-', '--', ':', '-.')
		def formatter(it):
			""" Formats the product correctly for use by pyplot.
			"""
			for line, point, color in it:
				yield color, '{}{}'.format(point, line)
		return formatter(
			itertools.cycle(itertools.product(lines, ['o'], colors))
		)

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

		stats = log.enumerate_statistics()
		validation_data = {}
		for stat in stats:
			if stat.data_type is not Statistic.Type.VALIDATION:
				continue
			if stat.name != 'total':
				continue
			if not stat.tags or stat.tags[0] != 'loss':
				continue
			vbatch, vtime, vloss = log.load_statistic(stat)
			if all(x is None for x in (vbatch, vtime, vloss)):
				logger.warning('Something is wrong with our statistics '
					'enumeration. Skipping: %s', stat)
				continue
			k = stat.tags[1] if len(stat.tags) > 1 else ''
			validation_data[k] = {
				'batch' : vbatch,
				'time' : vtime,
				'loss' : vloss
			}
		validation_data = OrderedDict(sorted(validation_data.items()))
		if len(validation_data) == 2 and 'default' in validation_data:
			validation_data = {'' : validation_data['']}

		def format_title(k):
			if not k:
				if len(validation_data) > 1:
					k = 'Overall'
				else:
					return ''
			return ': {}'.format(k)

		path = self.plots.get('loss_per_batch')
		if path:
			plt.xlabel('Batch')
			plt.ylabel('Loss')

			if batch is None:
				batch = numpy.arange(1, len(loss)+1)
			t_line, = plt.plot(batch, loss, 'co-', label='Training Loss')

			v_lines = []
			for (color, style), (k, data) in zip(
				self.validation_style(),
				validation_data.items()
			):
				if data['batch'] is None:
					data['batch'] = numpy.arange(1, len(data['loss'])+1)
				v_line, = plt.plot(
					data['batch'],
					data['loss'],
					style,
					label='Validation Loss{}'.format(format_title(k)),
					color=color
				)
				v_lines.append(v_line)

			plt.legend(
				handles=[t_line] + v_lines,
				fontsize=6,
				markerscale=0.5
			)
			plt.savefig(path, transparent=True, bbox_inches='tight')
			plt.clf()

			logger.debug('Loss-per-batch plot saved to: %s', path)

		path = self.plots.get('loss_per_time')
		if path:
			plt.xlabel('Time')
			plt.ylabel('Loss')

			t_line = []
			if time is None:
				logger.warning('No training timestamps available for '
					'loss-per-time plot.')
			else:
				t_line, = plt.plot(time, loss, 'co-', label='Training Loss')
				t_line = [t_line]

			v_lines = []
			for (color, style), (k, data) in zip(
				self.validation_style(),
				validation_data.items()
			):
				if data['time'] is None:
					logger.warning('No validation timestamps available for '
						'loss-per-time plot with provider "%s".', k)
					continue
				v_line, = plt.plot(
					data['time'],
					data['loss'],
					style,
					label='Validation Loss{}'.format(format_title(k)),
					color=color
				)
				v_lines.append(v_line)

			plt.legend(
				handles=t_line+v_lines,
				fontsize=6,
				markerscale=0.5
			)
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

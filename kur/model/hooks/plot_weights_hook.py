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

import itertools
from collections import OrderedDict

import numpy
import tempfile

from . import TrainingHook
from ...loggers import PersistentLogger, Statistic

import logging
import matplotlib.pyplot as plt
import numpy as np
import math
logger = logging.getLogger(__name__)
from ...utils import DisableLogging, idx

###############################################################################
class PlotWeightsHook(TrainingHook):
	""" Hook for creating plots of loss.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the hook.
		"""
		return 'plot_weights'

	###########################################################################
	def __init__(self, plot_directory, weight_file, with_weights, plot_every_n_epochs, *args, **kwargs):
		""" Creates a new plot hook for plotting weights of layers
		"""

		super().__init__(*args, **kwargs)

		self.directory = plot_directory
		if not os.path.exists(self.directory):
			os.makedirs(self.directory)

		self.plot_every_n_epochs = plot_every_n_epochs

		if weight_file is None:
			self.weight_file = None
		else:
			self.weight_file = weight_file

		self.with_weights = with_weights

		try:
			import matplotlib					# pylint: disable=import-error
		except:
			logger.exception('Failed to import "matplotlib". Make sure it is '
				'installed, and if you have continued trouble, please check '
				'out our troubleshooting page: https://kur.deepgram.com/'
				'troubleshooting.html#plotting')
			raise

		# Set the matplotlib backend to one of the known backends.
		matplotlib.use('Agg')

	###########################################################################
	def notify(self, status, log=None, info=None, model=None):
		""" Creates the plot.
		"""

		from matplotlib import pyplot as plt	# pylint: disable=import-error



		if status not in (
			# the plotting is allowed only at end of epoch
			TrainingHook.EPOCH_END,
		):

			return


		weight_path = None
		tempdir = tempfile.mkdtemp()
		weight_path = os.path.join(tempdir, 'current_epoch_model')
		model.save(weight_path)

		def plot_weights(kernel_filename):

			filename_cut_dir = kernel_filename[kernel_filename.find("dense") :]

			w = idx.load(kernel_filename)

			w_min = np.min(w)
			w_max = np.max(w)


			s1, s2 = w.shape
			if s1 < s2:
				w = w.reshape((s2, s1))

			flattend_pixels, num_classes = w.shape
			num_grids = math.ceil(math.sqrt(num_classes))
			width_pixels = math.ceil(math.sqrt(flattend_pixels))

			fig, axes = plt.subplots(num_grids, num_grids)
			fig.subplots_adjust(hspace=0.3, wspace=0.3)

			for i, ax in enumerate(axes.flat):
				if i<num_classes:

					try:
						image = w[:, i].reshape((width_pixels, width_pixels))
					except ValueError:
						logger.error("\nweights ({}), its first dim must be a square of an positive integer, but current weights first dim is {}. So no plotting for this weights.\n\n".format(filename_cut_dir, w.shape[0]))
						return

					ax.set_xlabel("W_column: {0}".format(i))
					ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

				if i == 0:
					ax.set_title("validation_loss: {}".format(round(info['Validation loss'][None]['labels'], 3)))

				# Remove ticks from each sub-plot.
				ax.set_xticks([])
				ax.set_yticks([])

			# either display or save the plot
			# plt.show()
			plt.savefig('{}/{}_epoch_{}.png'.format(self.directory, filename_cut_dir, info['epoch']))


		def plot_conv_weights(kernel_filename, input_channel=0):
			w = idx.load(kernel_filename)

			w_min = np.min(w)
			w_max = np.max(w)

			s1, s2, s3, s4 = w.shape
			if s1 > s4:
				w = w.reshape((s3, s4, s2, s1))

			num_filters = w.shape[3]
			num_grids = math.ceil(math.sqrt(num_filters))

			fig, axes = plt.subplots(num_grids, num_grids)
			for i, ax in enumerate(axes.flat):
				if i<num_filters:

					img = w[:, :, input_channel, i]
					ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

				if i == 0:
					ax.set_title("validation_loss: {}".format(round(info['Validation loss'][None]['labels'], 3)))

				ax.set_xticks([])
				ax.set_yticks([])

			filename_cut_dir = kernel_filename[kernel_filename.find("convol") :]

			plt.savefig('{}/{}_epoch_{}.png'.format(self.directory, filename_cut_dir, info['epoch']))



		if info['epoch'] == 1 or info['epoch'] % self.plot_every_n_epochs == 0:
			valid_weights_filenames = []

			if self.weight_file is None:
				self.weight_file = weight_path

			for dirpath, _, filenames in os.walk(self.weight_file):
				for this_file in filenames:
					valid_weights_filenames.append(dirpath+"/"+this_file)

			for this_file in valid_weights_filenames:
				for weight_keywords in self.with_weights:

					if this_file.find(weight_keywords[0]) > -1 and this_file.find(weight_keywords[1]) > -1:

						if weight_keywords[0].find("convol") > -1 or weight_keywords[1].find("convol") > -1:

							plot_conv_weights(this_file)

						else:
							plot_weights(this_file)

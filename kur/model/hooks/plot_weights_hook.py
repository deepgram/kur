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
	def __init__(self, plot_directory, weight_file, weight_keywords1, weight_keywords2, plot_every_n_epochs, *args, **kwargs):
		""" Creates a new plot_weights hook, get weights filenames, path for saving plots, keywords for selecting layer-weights, num_epochs before plotting, and matplotlib ready.
		"""

		super().__init__(*args, **kwargs)

		self.directory = plot_directory
		if not os.path.exists(self.directory):
			os.makedirs(self.directory)

		# bring in kurfile: hooks: plot_weights: weight_file, weight_file_keywords
		self.plot_every_n_epochs = plot_every_n_epochs
		self.weight_file = weight_file
		self.weight_keywords1 = weight_keywords1
		self.weight_keywords2 = weight_keywords2
		# import matplotlib and use
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
	def notify(self, status, log=None, info=None):
		""" Creates the plot.
		"""

		from matplotlib import pyplot as plt	# pylint: disable=import-error

		# logger.critical('PlotWeightsHook received training message.')

		if status not in (
			# TrainingHook.TRAINING_END,
			# TrainingHook.VALIDATION_END,
			TrainingHook.EPOCH_END, # , is a must here
		):
			logger.critical('\n\nPlotWeightsHook is tried here, but it does not handle the specified status.\n\n')
			return


		# borrowed from https://hyp.is/MKzd7C4eEeeWlPvso_EWdg/nbviewer.jupyter.org/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/01_Simple_Linear_Model.ipynb
		def plot_weights(kernel_filename):
			# designed to plot weights of a single dense layer model on recognising images of single color 

			# load weights from weight files in idx format
			w = idx.load(kernel_filename)

			# Get the lowest and highest values for the weights.
			# This is used to correct the colour intensity across
			# the images so they can be compared with each other.
			w_min = np.min(w)
			w_max = np.max(w)

			num_classes = w.shape[-1]
			flattend_pixels = w.shape[0]

			# Number of grids to plot.
			# Rounded-up, square-root of the number of filters.
			num_grids = math.ceil(math.sqrt(num_classes))
			width_pixels = math.ceil(math.sqrt(flattend_pixels))

			# Create figure with a grid of sub-plots.
			fig, axes = plt.subplots(num_grids, num_grids)
			# Create figure with 3x4 sub-plots,
			# where the last 2 sub-plots are unused.
			# fig, axes = plt.subplots(3, 4)
			fig.subplots_adjust(hspace=0.3, wspace=0.3)


			for i, ax in enumerate(axes.flat):
				# Only use the weights for the first 10 sub-plots.
				if i<num_classes:
				# if i<64:
					# Get the weights for the i'th digit and reshape it.
					# Note that w.shape == (img_size_flat, 10)
					# mnist (28, 28)
					# cifar (32,32,3)
					# image = w[:, i].reshape((28, 28))
					image = w[:, i].reshape((width_pixels, width_pixels))


					# Set the label for the sub-plot.
					ax.set_xlabel("Weights: {0}".format(i))


					# Plot the image.
					ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

				if i == 0:
					# how to make a title for plotting
					ax.set_title("validation_loss: {}".format(round(info['Validation loss'][None]['labels'], 3)))

				# Remove ticks from each sub-plot.
				ax.set_xticks([])
				ax.set_yticks([])
			# if we plot while training, we can't save it
			# plt.show()

			# get filename without "dir/.."
			filename_cut_dir = kernel_filename[kernel_filename.find("/..")+3 :]
			# save figure with a nicer name
			plt.savefig('{}/{}_epoch_{}.png'.format(self.directory, filename_cut_dir, info['epoch']))

		# borrowed from  https://hyp.is/4mtFzjBSEeeNikfkfV9o4w/nbviewer.jupyter.org/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
		def plot_conv_weights(kernel_filename, input_channel=0):

			# load weights from weight files in idx format
			w = idx.load(kernel_filename)

			# Get the lowest and highest values for the weights.
			# This is used to correct the colour intensity across
			# the images so they can be compared with each other.
			w_min = np.min(w)
			w_max = np.max(w)

			# Number of filters used in the conv. layer.
			num_filters = w.shape[3]

			# Number of grids to plot.
			# Rounded-up, square-root of the number of filters.
			num_grids = math.ceil(math.sqrt(num_filters))

			# Create figure with a grid of sub-plots.
			fig, axes = plt.subplots(num_grids, num_grids)

			# Plot all the filter-weights.
			for i, ax in enumerate(axes.flat):
				# Only plot the valid filter-weights.
				if i<num_filters:
					# Get the weights for the i'th filter of the input channel.
					# See new_conv_layer() for details on the format
					# of this 4-dim tensor.
					img = w[:, :, input_channel, i]

					# Plot image.
					ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

				# Remove ticks from the plot.
				ax.set_xticks([])
				ax.set_yticks([])

		    # Ensure the plot is shown correctly with multiple plots
		    # in a single Notebook cell.
		    # if we plot while training, we can't save it
			# plt.show()

			# get filename without "dir/.."
			filename_cut_dir = kernel_filename[kernel_filename.find("/..")+3 :]
			# save figure with a nicer name
			plt.savefig('{}/{}_epoch_{}.png'.format(self.directory, filename_cut_dir, info['epoch']))



		if info['epoch'] == 1 or info['epoch'] % self.plot_every_n_epochs == 0:
			# save weights plots
			logger.critical("\n\nLet's print weights at epoch idx 1 or every %s epochs\n\n", self.plot_every_n_epochs)


			# get all the validation weights names
			valid_weights_filenames = []
			# how to give a path name to plot_weights???
			for dirpath, _, filenames in os.walk(self.weight_file): # mnist or cifar
				for this_file in filenames:
					valid_weights_filenames.append(dirpath+"/"+this_file)

			# find two layers-weights with selected keywords, and plot their weights, either single dense layer model or covolutional layer weights
			for this_file in valid_weights_filenames:
				if this_file.find(self.weight_keywords1[0]) > -1 and this_file.find(self.weight_keywords1[1]) > -1:


					if self.weight_keywords1[0].find("convol") > -1 or self.weight_keywords1[1].find("convol") > -1:
						plot_conv_weights(this_file)
					else:
						plot_weights(this_file)

				if this_file.find(self.weight_keywords2[0]) > -1 and this_file.find(self.weight_keywords2[1]) > -1:


					if self.weight_keywords2[0].find("convol") > -1 or self.weight_keywords2[1].find("convol") > -1:
						plot_conv_weights(this_file)
					else:
						plot_weights(this_file)
			# save validation_loss on the plotting


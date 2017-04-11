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

import numpy

from . import Loss
from ..sources import RepeatSource, DerivedSource
from ..engine import PassthroughEngine
from ..utils import can_import

logger = logging.getLogger(__name__)

###############################################################################
class ScaledSource(DerivedSource):
	""" Derived source which scales `input_length` by the same
		amount that the length of `scale_this` is scaled down
		to `target`.
	"""
	###########################################################################
	def __init__(self, model, relative_to, to_this, scale_this):
		super().__init__()
		self.model = model
		self.relative_to = relative_to
		self.to_this = to_this
		self.scale_this = scale_this
		self.normal_shape = None

	###############################################################
	def derive(self, inputs):
		# Break it apart
		sizes, = inputs
		if sizes.ndim < 2:
			sizes = numpy.expand_dims(sizes, -1)
		outputs = numpy.array(
			[
				self.model.get_shape_at_layer(
					name=self.to_this,
					assumptions={
						self.relative_to : \
							(x[0], ) + tuple(self.normal_shape[1:])
					}
				)[0]
				for x in sizes
			],
			dtype='int32'
		)
		return numpy.expand_dims(outputs, axis=1)

	###############################################################
	def setup(self):
		self.normal_shape = self.model.get_shape_at_layer(
			name=self.relative_to
		)

	###############################################################
	def shape(self):
		return (None, 1)

	###############################################################
	def requires(self):
		return (self.scale_this, )

###################################################################
class FlattenSource(DerivedSource):
	""" Derived source which converts a rectangular array of
		labels (one for each sample, and all padded to the same
		width), along with the label lengths (the number of non-
		padded entries in each sample) and produces a single, long,
		flattened list of labels.
	"""

	###############################################################
	def __init__(self, label, label_length):
		super().__init__()
		self.label = label
		self.label_length = label_length

	###############################################################
	def derive(self, inputs):
		# Break it apart
		labels, label_lengths = inputs
		outputs = numpy.array([
			x
			for label, label_length in zip(labels, label_lengths)
			for x in label[:label_length[0]]
		])
		# Warp-CTC is strange: it uses a variable-length vector of
		# flattened labels. That doesn't look much like a deep
		# learning tensor! So we can cheat without imposing much of
		# a memory hit.
		outputs = numpy.lib.stride_tricks.as_strided(
			outputs,
			shape=(len(labels), ) + outputs.shape,
			strides=(0, ) + outputs.strides
		)
		return outputs

	###############################################################
	def shape(self):
		return (None,)

	###############################################################
	def requires(self):
		return (self.label, self.label_length)

###############################################################################
class Ctc(Loss):
	""" Connectionist Temporal Classification loss function
	"""

	KNOWN_VARIANTS = {'warp', 'loss_scale'}

	###########################################################################
	def __init__(self, input_length, output_length, output, relative_to=None,
		variant=None, **kwargs):
		""" Creates a new CTC loss function.

			# Arguments
		"""
		super().__init__(**kwargs)

		if variant is None:
			variant = set()
		elif isinstance(variant, str):
			variant = {variant}
		elif isinstance(variant, (list, tuple, set)):
			variant = set(variant)
		else:
			raise ValueError('Unexpected or unsupport CTC variant type: {}'
				.format(variant))

		for x in variant:
			if x not in Ctc.KNOWN_VARIANTS:
				logger.warning('Ignoring an unknown variant to the CTC loss '
					'function: %s', x)

		self.variant = variant

		self.input_length = input_length
		self.output_length = output_length
		self.output = output
		self.relative_to = relative_to

	###########################################################################
	def get_loss(self, model, target, output):
		""" Returns the loss function that can be used by the implementation-
			specific model.
		"""
		backend = model.get_backend()

		if backend.get_name() == 'keras':

			import keras.backend as K

			if 'warp' in self.variant:

				# Just use the built-in Keras CTC loss function.
				logger.info('Attaching Warp-CTC loss function to model '
					'output "%s".', target)

				if backend.get_toolchain() != 'theano':
					logger.error('If you want to use warp-ctc, you need to '
						'use the Theano backend to Keras.')
					raise ValueError('Warp-CTC is currently only supported '
						'with the Theano backend to Keras.')

			else:
				# Just use the built-in Keras CTC loss function.
				logger.debug('Attaching built-in Keras CTC loss function to '
					'model output "%s".', target)

			ctc_scaled = 'ctc_scaled_{}'.format(self.input_length)
			flattened_labels = 'ctc_flattened_labels_{}'.format(target)

			transcript_length = K.placeholder(
				ndim=2,
				dtype='int32',
				name=self.output_length
			)
			transcript = K.placeholder(
				ndim=2,
				dtype='int32',
				name=flattened_labels if 'warp' in self.variant \
					else self.output
			)
			utterance_length = K.placeholder(
				ndim=2,
				dtype='int32',
				name=self.input_length if self.relative_to is None \
					else ctc_scaled
			)

			if self.relative_to is not None:
				model.add_data_source(
					ctc_scaled,
					ScaledSource(
						model,
						relative_to=self.relative_to,
						to_this=target,
						scale_this=self.input_length
					)
				)

			if 'warp' in self.variant:
				model.add_data_source(
					flattened_labels,
					FlattenSource(
						self.output,
						self.output_length
					)
				)

				try:
					import ctc					# pylint: disable=import-error
				except ImportError:
					logger.error('The warp-CTC loss function was requested,  '
						'but we cannot find the "ctc" library. See our '
						'troubleshooting page for helpful tips.')
					raise ImportError('Cannot find the "ctc" library, which '
						'is needed when using the "warp" variant of the CTC '
						'loss function.')

				out = ctc.cpu_ctc_th(
					output.dimshuffle((1, 0, 2)),
					K.squeeze(utterance_length, -1),
					transcript[0]+1,
					K.squeeze(transcript_length, -1)
				)
			else:
				out = K.ctc_batch_cost(
					transcript,
					output,
					utterance_length,
					transcript_length
				)

			if 'loss_scale' in self.variant:
				logger.debug('Loss scaling is active.')
				out = out * K.mean(
					K.cast(utterance_length, K.dtype(out))
				) / 100

			return (
				(
					(self.output_length, transcript_length),
					(flattened_labels if 'warp' in self.variant \
						else self.output, transcript),
					(self.input_length if self.relative_to is None \
						else ctc_scaled, utterance_length)
				),
				out
			)

		elif backend.get_name() == 'pytorch':

			if 'warp' not in self.variant:
				logger.error('PyTorch does not include a native CTC loss '
					'function yet. However, PyTorch bindings to Warp CTC are '
					'available (SeanNaren/warp-ctc). Try installing that, and '
					'then settings variant=warp.')
				raise ValueError('Only Warp CTC is supported for PyTorch '
					'right now.')

			ctc_scaled = 'ctc_scaled_{}'.format(self.input_length)
			flattened_labels = 'ctc_flattened_labels_{}'.format(target)
			transcript_length = model.data.placeholder(
				self.output_length,
				location='cpu',
				data_type='int'
			)
			transcript = model.data.placeholder(
				flattened_labels,
				location='cpu',
				data_type='int'
			)
			utterance_length = model.data.placeholder(
				self.input_length if self.relative_to is None else ctc_scaled,
				location='cpu',
				data_type='int'
			)

			if self.relative_to is not None:
				model.add_data_source(
					ctc_scaled,
					ScaledSource(
						model,
						relative_to=self.relative_to,
						to_this=target,
						scale_this=self.input_length
					)
				)

			if 'warp' in self.variant:
				model.add_data_source(
					flattened_labels,
					FlattenSource(
						self.output,
						self.output_length
					)
				)

			try:
				from warpctc_pytorch import CTCLoss	# pytorch: disable=import-error
			except ImportError:
				logger.error('The warp-CTC loss function was requested,  '
					'but we cannot find the "warpctc_pytorch" library. See '
					'out troubleshooting page for helpful tips.')
				raise ImportError('Cannot find the "warpctc_pytorch" library, '
					'which is needed when using the "warp" variant of the CTC '
					'loss function.')

			loss = model.data.move(CTCLoss())

			def basic_ctc_loss(inputs, output):
				""" Computes CTC loss.
				"""
				return loss(
					output.transpose(1, 0).contiguous(),
					inputs[0][0]+1,		# transcript[0]+1
					inputs[1].squeeze(1),	# K.squeeze(utterance_length, -1),
					inputs[2].squeeze(1)	# K.squeeze(transcript_length, -1)
				) / output.size(0)

			if 'loss_scale' in self.variant:
				logger.debug('Loss scaling is active.')

				def loss_scale(inputs, output):
					""" Computes CTC loss.
					"""
					return basic_ctc_loss(inputs, output) * \
						(inputs[1].float().mean() / 100)

				get_ctc_loss = loss_scale
			else:
				get_ctc_loss = basic_ctc_loss

			return [
				[
					(flattened_labels if 'warp' in self.variant \
						else self.output, transcript),
					(self.input_length if self.relative_to is None \
						else ctc_scaled, utterance_length),
					(self.output_length, transcript_length)
				],
				get_ctc_loss
			]

		else:
			raise ValueError('Unsupported backend "{}" for loss function "{}"'
				.format(backend.get_name(), self.get_name()))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

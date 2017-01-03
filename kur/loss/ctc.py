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

logger = logging.getLogger(__name__)

###############################################################################
class Ctc(Loss):
	""" Connectionist Temporal Classification loss function
	"""

	###########################################################################
	def __init__(self, input_length, output_length, output, relative_to=None,
		variant=None, **kwargs):
		""" Creates a new CTC loss function.

			# Arguments
		"""
		super().__init__(**kwargs)

		if variant is None:
			self.variant = None
		elif variant == 'warp':
			self.variant = 'warp'
		else:
			raise ValueError('Unsupported CTC variant: {}'.format(variant))

		self.input_length = input_length
		self.output_length = output_length
		self.output = output
		self.relative_to = relative_to

	###########################################################################
	def get_loss(self, backend):
		""" Returns the loss function that can be used by the implementation-
			specific model.
		"""
		if backend.get_name() == 'keras':

			if self.variant is None:
				return lambda y_true, y_pred: y_pred
			elif self.variant == 'warp':
				raise NotImplementedError
			else:
				raise ValueError('Unsupported variant "{}" on loss function '
					'"{}" for backend "{}".'.format(self.variant,
						self.get_name(), backend.get_name()))

		else:
			raise ValueError('Unsupported backend "{}" for loss function "{}"'
				.format(backend.get_name(), self.get_name()))

	###########################################################################
	def modify(self, model, name):
		""" Modify/extend the model to fit this loss function.

			Some loss functions will want to modify the model in some way in
			order to properly instrument the model. For example, CTC loss is a
			little different in the sense that it wants additional inputs at
			training and evaluation time. Many loss functions, however, will
			not need this functionality.

			# Arguments

			model: Model instance. The model to modify.
			name: str. The output of the model to modify.

			# Return value

			The name of the model output to apply this loss function to.
			Normally, this is the same as `name`, but if the loss function
			modifies the model, then it may need to act on a different layer
			instead.
		"""
		backend = model.backend
		if backend.get_name() == 'keras':

			if self.variant is None:

				# Just use the built-in Keras CTC loss function.
				logger.debug('Attaching built-in Keras CTC loss function to '
					'model output "%s".', name)

				# pylint: disable=import-error
				import keras.backend as K
				import keras.layers as L
				# pylint: enable=import-error

				from ..containers.layers import Layer, Placeholder

				ctc_name = 'ctc_{}'.format(name)

				###############################################################
				class CtcLayer(Layer):
					""" Layer implementation of CTC
					"""
					###########################################################
					@classmethod
					def get_container_name(cls):
						""" Returns the name of the container class.
						"""
						return 'ctc_layer'
					###########################################################
					def _parse_pre(self, engine):
						""" Pre-parsing hook.
						"""
						super()._parse_pre(engine)
						self.name = ctc_name
					###########################################################
					def _build(self, model):
						""" Builds the CTC layers
						"""
						yield L.Lambda(
							CtcLayer.ctc_lambda_func,
							output_shape=(1,),
							name=self.name
						)
					###########################################################
					@staticmethod
					def ctc_lambda_func(args):
						""" Wrapper for the actual CTC loss function.
						"""
						y_pred, labels, input_length, label_length = args
						return K.ctc_batch_cost(
							labels,			# True output
							y_pred,			# Model output
							input_length,	# CTC input length
							label_length	# CTC output length
						)
					###########################################################
					def is_anonymous(self):
						""" Whether or not this container is intended to be
							used by end-users.
						"""
						return True
					###########################################################
					def shape(self, input_shapes):
						return ()

				###############################################################
				class ScaledSource(DerivedSource):
					""" Derived source which scales `input_length` by the same
						amount that the length of `scale_this` is scaled down
						to `target`.
					"""
					###########################################################
					def __init__(self, model, relative_to, to_this,
						scale_this):
						super().__init__()
						self.model = model
						self.relative_to = relative_to
						self.to_this = to_this
						self.scale_this = scale_this
						self.normal_shape = None
					###########################################################
					def derive(self, inputs):
						# Break it apart
						sizes, = inputs
						outputs = numpy.array(
							[
								self.model.get_shape_at_layer(
									name=self.to_this,
									assumptions={
										self.relative_to : \
											(x[0], ) + self.normal_shape[1:]
									}
								)[0]
								for x in sizes
							],
							dtype='int32'
						)
						return numpy.expand_dims(outputs, axis=1)
					###########################################################
					def setup(self):
						self.normal_shape = self.model.get_shape_at_layer(
							name=self.relative_to
						)
					###########################################################
					def shape(self):
						return (1, )
					###########################################################
					def requires(self):
						return (self.scale_this, )

				# Determine what the scaled output should be called.
				repeat = 1
				while True:
					ctc_scaled = 'ctc_scaled_{}_{}'.format(
						self.input_length,
						repeat
					)
					if not model.has_data_source(ctc_scaled):
						break
					repeat += 1

				# Add the new layers
				new_containers = [
					Placeholder({
						'input' : {
							'type' : 'int32',
							'shape' : (1,)
						},
						'name' : ctc_scaled if self.relative_to is not None \
							else self.input_length
					}),
					Placeholder({
						'input' : {
							'type' : 'int32',
							'shape' : (1,)
						},
						'name' : self.output_length
					}),
					Placeholder({'input' : self.output}),
					CtcLayer({'inputs' : [
						name,
						self.output,
						ctc_scaled if self.relative_to is not None \
							else self.input_length,
						self.output_length
					]})
				]

				# In case the CTC'd layer got labeled an output, we need to
				# remove the `sink` designation.
				model.root.get_relative_by_name(name).sink = False

				# Parse out the containers
				engine = PassthroughEngine()
				for container in new_containers:
					container.parse(engine)

				# Add these new containers to the model.
				model.extend(ctc_name, new_containers)

				# Add the additional data sources.
				model.add_data_source(ctc_name, RepeatSource(0.))
				if self.relative_to is not None:
					model.add_data_source(
						ctc_scaled,
						ScaledSource(
							model,
							relative_to=self.relative_to,
							to_this=name,
							scale_this=self.input_length
						)
					)

				return ctc_name

			elif self.variant == 'warp':
				raise NotImplementedError
			else:
				raise ValueError('Unsupported variant "{}" on loss function '
					'"{}" for backend "{}".'.format(self.variant,
						self.get_name(), backend.get_name()))
		else:
			raise ValueError('Unsupported backend "{}" for loss function "{}"'
				.format(backend.get_name(), self.get_name()))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

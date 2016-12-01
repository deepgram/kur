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

import contextlib
import io
import re
import sys
import logging
import numpy
from . import Backend
from ..containers import Layer
from ..loss import Loss
from ..model import ExtensionState
from ..utils import can_import, EnvironmentalVariable, redirect_stderr

logger = logging.getLogger(__name__)

################################################################################
class KerasBackend(Backend):
	""" A Keras backend.

		# Dependencies

		- keras
		- theano OR tensorflow
		- h5py
	"""

	############################################################################
	@classmethod
	def is_supported(cls):
		""" Returns True if this backend can be used.
		"""
		return can_import('keras') and (
			can_import('theano') or can_import('tensorflow')
		)

	############################################################################
	def __init__(self, backend=None, *args, **kwargs):
		""" Creates a new Keras backend.

			As per the base class documentation, we should do all necessary
			Keras-related initialization here, including checking obvious things
			like "Is Keras installed?" or "Is a backend installed?"

			# Arguments

			backend: str or None (default: None). The Keras backend to use
				(either "theano" or "tensorflow"). None uses the system default.
		"""

		super().__init__(*args, **kwargs)

		if backend is not None:
			if 'keras' in sys.modules:
				import keras.backend as K		# pylint: disable=import-error
				if K.backend() != backend:
					logger.warning('Keras was already imported by the time the '
						'Kur backend was instantiated. Kur was asked to use '
						'Keras %s backend, but Keras is already using %s. We '
						'cannot change the Keras backend at this point, so we '
						'will try to work with the currently loaded backend. '
						'In the future, try to let Kur manage importing Keras.',
						backend, K.backend()
					)

		# Make sure Keras is loaded.
		# Now, Keras always prints out a "Using {Theano|TensorFlow} backend."
		# statement that is frankly unbecoming. So we'll just gobble it up here.
		x = io.StringIO()
		with redirect_stderr(x):

			with EnvironmentalVariable(KERAS_BACKEND=backend):
				import keras	# pylint: disable=import-error,unused-variable

		# And now we can set the dimension ordering.
		keras.backend.set_image_dim_ordering('tf')

	############################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the backend class.

			This is used by the base class's factory method.
		"""
		return 'keras'

	############################################################################
	def connect(self, inputs, target):
		""" Use the Keras functional API to connect to layers

			# Notes:

			- You will need input placeholders in place before doing this,
			  otherwise Keras's shape-checking will fail.
		"""
		if isinstance(inputs, Layer):
			inputs = [inputs]
		return target(inputs)

	############################################################################
	def save(self, model, filename):
		""" Saves the model weights to the given filename.
		"""
		import keras.models as M				# pylint: disable=import-error

		result = M.Model(
			input=[node.value for node in model.inputs.values()],
			output=[node.value for node in model.outputs.values()]
		)
		result.save_weights(filename)

	############################################################################
	def restore(self, model, filename):
		""" Load the model weights from the given filename.
		"""
		import keras.models as M				# pylint: disable=import-error

		result = M.Model(
			input=[node.value for node in model.inputs.values()],
			output=[node.value for node in model.outputs.values()]
		)
		result.load_weights(filename, by_name=True)

	############################################################################
	def compile(self, model, loss=None, optimizer=None):
		""" Returns the Keras model instance.
		"""

		with ExtensionState(model):

			if loss is not None:
				loss, alias = self._apply_loss(model, loss)
				rev = {v : k for k, v in alias.items()}
			else:
				loss, alias, rev = None, {}, {}

			import keras.models as M			# pylint: disable=import-error
			logger.debug('Instantiating a Keras model.')
			result = M.Model(
				input=[node.value for node in model.inputs.values()],
				output=[node.value for node in model.outputs.values()]
			)

			if logger.isEnabledFor(logging.DEBUG):
				x = io.StringIO()
				with contextlib.redirect_stdout(x):
					result.summary()
				for line in x.getvalue().split('\n'):
					logger.debug(line)

			if loss is not None:# and optimizer is not None:
				logger.debug('Starting to compile the Keras model.')
				result.compile(
					loss={name : func.get_loss(self)
						for name, func in loss.items()},
					optimizer=optimizer.get_optimizer(self) \
						if optimizer is not None else None,
					loss_weights={name : func.get_weight()
						for name, func in loss.items()}
				)

			return {'model' : result, 'alias' : alias, 'rev_alias' : rev}

	############################################################################
	def _apply_loss(self, model, loss=None):	# pylint: disable=no-self-use
		""" Applies the loss functions to the model.

			# Arguments

			# Return value
		"""
		# How many outputs are in this model?
		num_outputs = len(model.outputs)

		if isinstance(loss, Loss):
			loss = [loss]

		if isinstance(loss, (list, tuple)):
			if len(loss) == 1:
				while len(loss) < num_outputs:
					loss.append(loss[0])
			elif len(loss) != num_outputs:
				raise ValueError('Wrong number of loss functions specified. '
					'There must be as many loss functions as model outputs, or '
					'a single loss function that can be replicated for each '
					'model output. In this case, there were {} loss functions '
					'given, but the model has {} outputs.'
						.format(len(loss), num_outputs))

			temp = {}
			for name, output_loss in zip(model.outputs, loss):
				temp[name] = output_loss
			loss = temp				# pylint: disable=redefined-variable-type

		elif not isinstance(loss, dict):
			raise ValueError('Unexpected form for the loss function. Expected '
				'a single loss function, a list of loss functions, or a '
				'dictionary mapping model outputs to loss functions. Instead '
				'we received this: {}'.format(loss))

		supplied_loss = set(loss.keys())
		required_loss = set(model.outputs.keys())

		for name in supplied_loss - required_loss:
			logger.warning('Loss function was supplied for "%s", but no such '
				'output exists in the model. Maybe you meant one of these: %s. '
				'Supplied loss functions: %s. Required loss functions: %s.',
				name,
				', '.join(x for x in required_loss - supplied_loss),
				', '.join(supplied_loss),
				', '.join(required_loss)
			)
		missing_loss = False
		for name in required_loss - supplied_loss:
			logger.error('Loss function is needed for model output "%s", but '
				'no such loss function was supplied. Maybe you meant one of '
				'these: %s. Supplied loss functions: %s. Required loss '
				'functions: %s.',
				name,
				', '.join(x for x in supplied_loss - required_loss),
				', '.join(supplied_loss),
				', '.join(required_loss)
			)
			missing_loss = True
		if missing_loss:
			raise ValueError('One or more model outputs did not have loss '
				'functions supplied. Loss functions are needed for these '
				'model outputs: {}'.format(
					', '.join(name for name in required_loss - supplied_loss)
			))

		# Once the model extension retracts, the alias keys still exist as the
		# Model inputs/outputs, and they will match data source names.
		# But the Keras model has inputs/outputs named after `alias[name]`.
		alias = {name : func.modify(model, name) for name, func in loss.items()}

		return loss, alias

	############################################################################
	def train(self, model, data, compiled):
		""" Fits the given model on a batch of data.
		"""
		metrics = compiled['model'].train_on_batch(
			{compiled['alias'].get(name, name) : data[name]
				for name in model.inputs},
			{compiled['alias'].get(name, name) : data[name]
				for name in model.outputs}
		)

		return KerasBackend._convert_metrics(
			metrics,
			compiled['model'].metrics_names,
			compiled['rev_alias'],
			model.outputs
		)

	############################################################################
	def test(self, model, data, compiled):
		""" Calculates the model loss on a batch of data.
		"""
		metrics = compiled['model'].test_on_batch(
			{compiled['alias'].get(name, name) : data[name]
				for name in model.inputs},
			{compiled['alias'].get(name, name) : data[name]
				for name in model.outputs}
		)

		return KerasBackend._convert_metrics(
			metrics,
			compiled['model'].metrics_names,
			compiled['rev_alias'],
			model.outputs
		)

	############################################################################
	@staticmethod
	def _convert_metrics(metrics, metrics_names, rev_alias, outputs):
		""" Formats the Keras metrics properly for use by Kur.
		"""
		if not isinstance(metrics, list):
			metrics = [metrics]
		un_numpy = lambda x: x.tolist() if isinstance(x, numpy.ndarray) else x
		metrics = {k : un_numpy(v) for k, v in zip(metrics_names, metrics)}

		loss = {}
		for k, v in metrics.items():
			match = re.match(r'^(.*)_loss$', k)
			if match:
				k = match.group(1)
				k = rev_alias.get(k, k)
				loss[k] = float(v)
		if not loss:
			for k in outputs:
				break
			loss[k] = metrics['loss']

		return loss

	############################################################################
	def evaluate(self, model, data, compiled):
		""" Evaluates the model on a batch of data.
		"""
		# Returns an array of model outputs, with one entry per branch.
		results = compiled['model'].predict_on_batch(
			{compiled['alias'].get(name, name) : data[name]
				for name in model.inputs}
		)

		if len(model.outputs) == 1:
			results = [results]

		return {name : result for name, result in zip(model.outputs, results)}

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

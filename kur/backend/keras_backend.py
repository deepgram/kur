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
import os
import sys
import tempfile
import shutil
import logging
import functools
from collections import OrderedDict
import numpy
from . import Backend
from .. import __homepage__
from ..loss import Loss
from ..utils import can_import, EnvironmentalVariable, redirect_stderr, idx
from ..providers import BatchProvider

logger = logging.getLogger(__name__)

###############################################################################
class KerasBackend(Backend):
	""" A Keras backend.

		# Dependencies

		- keras
		- theano OR tensorflow
		- h5py
	"""

	###########################################################################
	@classmethod
	def is_supported(cls):
		""" Returns True if this backend can be used.
		"""
		return can_import('keras') and (
			can_import('theano') or can_import('tensorflow')
		)

	###########################################################################
	def __init__(self, backend=None, optimizer=None, theano_flags=None,
		*args, **kwargs):
		""" Creates a new Keras backend.

			As per the base class documentation, we should do all necessary
			Keras-related initialization here, including checking obvious
			things like "Is Keras installed?" or "Is a backend installed?"

			# Arguments

			backend: str or None (default: None). The Keras backend to use
				(either "theano" or "tensorflow"). None uses the system
				default.
			optimizer: None or False (default: None). If False, Theano is told
				to disable optimizations. If None, Theano is not told to do
				anything special with optimization. This is supplied as a
				workaround to installing a BLAS library when training on the
				CPU. This is ignored for the TensorFlow backend.
		"""

		super().__init__(*args, **kwargs)

		if backend is not None:
			logger.info('The %s backend for Keras has been requested.',
				backend)

			if 'keras' in sys.modules:
				import keras.backend as K		# pylint: disable=import-error
				if K.backend() != backend:
					logger.warning('Keras was already imported by the time '
						'the Kur backend was instantiated. Kur was asked to '
						'use Keras %s backend, but Keras is already using %s. '
						'We cannot change the Keras backend at this point, so '
						'we will try to work with the currently loaded '
						'backend. In the future, try to let Kur manage '
						'importing Keras.', backend, K.backend())

			deps = {
				'theano' : ['theano'],
				'tensorflow' : ['tensorflow']
			}[backend]
			for dep in deps:
				if can_import(dep):
					continue
				if backend == 'tensorflow':
					logger.error('Your Kurfile is trying to use TensorFlow.')
					logger.error('However, we cannot find TensorFlow '
						'installed.')
					logger.error('At least it is easy to install!')
					logger.error('To install TensorFlow for CPU: pip install '
						'tensorflow')
					logger.error('To install TensorFlow for GPU: pip install '
						'tensorflow-gpu')
					logger.error('See our troubleshooting page for more '
						'information: %s', os.path.join(__homepage__,
						'troubleshooting.html'))
					raise ValueError('Need to install TensorFlow for this '
						'Kurfile to work.')
				else:
					logger.warning('The Keras backend was asked to use the %s '
						'backend, but %s does not appear to be installed. You '
						'will likely get an error about this soon.',
						backend, dep)

		else:
			logger.info('No particular backend for Keras has been requested.')
			if can_import('theano') and can_import('tensorflow'):
				logger.debug('Using the system-default Keras backend.')
			elif can_import('theano'):
				backend = 'theano'
				logger.debug('Only the Theano backend for Keras is installed, '
					'so we will try to use it.')
			elif can_import('tensorflow'):
				backend = 'tensorflow'
				logger.debug('Only the TensorFlow backend for Keras is '
					'installed, so we will try to use it.')
			else:
				logger.warning('No supported Keras backend seems to be '
					'installed. You will probably get an error about this '
					'shortly.')

		# Make sure Keras is loaded.
		# Now, Keras always prints out a "Using {Theano|TensorFlow} backend."
		# statement that is frankly unbecoming. So we'll just gobble it up here.
		x = io.StringIO()
		with redirect_stderr(x):

			env = {
				'KERAS_BACKEND' : backend,
				'THEANO_FLAGS' : os.environ.get('THEANO_FLAGS')
			}

			def replace_theano_flag(key, value):
				""" Updates the Theano flag variable.
				"""
				if env['THEANO_FLAGS']:
					parts = [i for i in env['THEANO_FLAGS'].split(',') \
						if not i.startswith('{}='.format(key))]
				else:
					parts = []
				parts.append('{}={}'.format(key, value))
				env['THEANO_FLAGS'] = ','.join(parts)

			if optimizer is False:
				logger.debug('Disabling the Theano optimizer.')
				replace_theano_flag('optimizer', 'None')

			if theano_flags is not None:
				for k, v in theano_flags.items():
					logger.debug('Setting Theano flag %s = %s', k, v)
					replace_theano_flag(k, v)

			if self.device is not None:
				replace_theano_flag('force_device', 'true')
				if self.device == 'cpu':
					logger.info('Forcing CPU.')
					replace_theano_flag('device', 'cpu')
					env['CUDA_VISIBLE_DEVICES'] = '100'
					logger.info('Requesting CPU')
				else:
					if self.device_number is None:
						replace_theano_flag('device', 'gpu')
						logger.info('Requesting any GPU')
					else:
						replace_theano_flag('device', 'gpu0')
						env['CUDA_VISIBLE_DEVICES'] = str(self.device_number)
						logger.info('Requesting GPU %d', self.device_number)

			# Supress the deluge of TensorFlow messages that we aren't
			# interested in.
			env['TF_CPP_MIN_LOG_LEVEL'] = '1'

			logger.debug('Overriding environmental variables: %s', env)
			EnvironmentalVariable(**env).push()

			import keras	# pylint: disable=import-error,unused-variable
			import keras.backend as K		# pylint: disable=import-error
			logger.info('Keras is loaded. The backend is: %s',
				K.backend())
			self.toolchain = K.backend()

		# And now we can set the dimension ordering.
		keras.backend.set_image_dim_ordering('tf')

		# The Keras `Wrapper` class accesses `Layer`'s `regularizers`
		# property (see `Wrapper.build()`), which triggers Keras' own
		# deprecation warning. Let's suppress this for now so that we don't
		# confuse our users.
		logging.getLogger('py.warnings').addFilter(
			type('theano_filter', (), {
				'filter' : lambda record: not (
					record.module == 'topology' and
					record.levelname == 'WARNING' and
					record.funcName == 'regularizers'
				)
			})
		)

	###########################################################################
	def get_toolchain(self):
		""" Returns a string describing the Keras backend being used, either
			'theano' or 'tensorflow'.
		"""
		return self.toolchain

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the backend class.

			This is used by the base class's factory method.
		"""
		return 'keras'

	###########################################################################
	def connect(self, inputs, target):
		""" Use the Keras functional API to connect to layers

			# Notes:

			- You will need input placeholders in place before doing this,
			  otherwise Keras's shape-checking will fail.
		"""
		if not isinstance(inputs, list):
			inputs = [inputs]
		return target(inputs)

	###########################################################################
	def save(self, model, filename):
		""" Saves the model weights to the given filename.
		"""
		import keras.models as M				# pylint: disable=import-error

		keras_model = M.Model(
			input=[node.value for node in model.inputs.values()],
			output=[node.value for node in model.outputs.values()]
		)

		self._save_keras(keras_model, filename)

	###########################################################################
	def _save_keras(self, keras_model, filename):
		""" Saves a native Keras model.
		"""
		path = os.path.expanduser(os.path.expandvars(filename))
		if os.path.exists(path):
			if not os.path.isdir(path):
				raise ValueError('Target weight exists, but it is not a '
					'directory. Kur expected a directory that it can work '
					'with. Please move or delete the existing path: {}'
					.format(path))

			for dirpath, _, filenames in os.walk(path):
				for this_file in filenames:
					if this_file.endswith('.kur'):
						os.unlink(os.path.join(dirpath, this_file))
		else:
			os.makedirs(path, exist_ok=True)

		layers = keras_model.flattened_layers \
			if hasattr(keras_model, 'flattened_layers') else keras_model.layers
		for layer in layers:
			layer_name = layer.name

			symbolic_weights = layer.weights
			weight_names, weight_values = \
				self._get_weight_names_and_values_from_symbolic(
					symbolic_weights
				)

			for name, val in zip(weight_names, weight_values):
				target = os.path.join(
					path,
					'{}+{}.kur'.format(layer_name, name)
				)
				idx.save(target, val)

	###########################################################################
	def _get_weight_names_and_values_from_symbolic(self, symbolic_weights):
		import keras.backend as K				# pylint: disable=import-error
		weight_values = K.batch_get_value(symbolic_weights)
		weight_names = [
			(
				str(w.name) if hasattr(w, 'name') and w.name \
					else 'param_{}'.format(i)
			)
			for i, (w, val) in enumerate(
				zip(symbolic_weights, weight_values)
			)
		]
		return weight_names, weight_values

	###########################################################################
	def restore(self, model, filename):
		""" Load the model weights from the given filename.
		"""
		import keras.models as M				# pylint: disable=import-error

		keras_model = M.Model(
			input=[node.value for node in model.inputs.values()],
			output=[node.value for node in model.outputs.values()]
		)

		try:
			self._restore_keras(keras_model, filename)
		except:
			logger.exception('Failed to load previously-saved Keras model. '
				'Are you accidentally loading pre-existing weights from an '
				'incompatible model? Make sure that any weights on disk are '
				'actually associated with the model you are trying to load '
				'them into.')
			raise

	###########################################################################
	def _restore_keras(self, keras_model, filename):
		""" Loads a native Keras model.
		"""
		import keras.backend as K				# pylint: disable=import-error

		path = os.path.expanduser(os.path.expandvars(filename))
		if os.path.exists(path):
			if not os.path.isdir(path):
				raise ValueError('Target weight exists, but it is not a '
					'directory. Kur expected a directory that it can work '
					'with. Please move or delete the existing path: {}'
					.format(path))
		else:
			raise ValueError('Target weight directory does not exist: {}'
				.format(path))

		layers = keras_model.flattened_layers \
			if hasattr(keras_model, 'flattened_layers') else keras_model.layers

		# Get a map from "layer name" to "layer instance" in the current model.
		index = {}
		for layer in layers:
			if layer.name:
				index.setdefault(layer.name, []).append(layer)

		# Enumerate all of the saved tensors, organized like this:
		# tensors = {
		#	'layer_1_name' : {
		#		'weight_1_name'  : '/path/to/file',
		#		...,
		#	},
		#	...
		# }
		tensors = self.enumerate_saved_tensors(path)

		# We want to put (symbolic_weights, weight_values) tuples in this.
		weight_value_tuples = []

		# Loop over the available weights.
		for layer_name, weights in tensors.items():

			# Load the weights.
			# This maps weight names to numpy arrays.
			weights = {k : idx.load(v) for k, v in weights.items()}

			# Now assign all of the weights to their corresponding symbolic
			# weights. Loop over all layers which use this name.
			for layer in index.get(layer_name, []):

				# Get the symbolic weights.
				symbolic_weights = layer.weights
				if len(weights) != len(symbolic_weights):
					raise ValueError('Layer "%s" expected %d weights, but we '
						'found %d on disk.', layer_name, len(symbolic_weights),
						len(weights))

				# Get the associated names (so we know what order to assign the
				# weights in.
				weight_names, _ = \
					self._get_weight_names_and_values_from_symbolic(
						symbolic_weights
					)
				for i, name in enumerate(weight_names):
					weight_value_tuples.append((symbolic_weights[i], weights[name]))

		# Assign all the weights in one batch (for efficiency).
		K.batch_set_value(weight_value_tuples)

	###########################################################################
	def enumerate_saved_tensors(self, path):
		""" Enumerates saved tensors (weights).
		"""
		result = {}

		regex = re.compile(r'^(?P<layer>.*?)\+(?P<weight>.*?)\.kur$')
		for dirpath, dirnames, filenames in os.walk(path): # pylint: disable=unused-variable
			for filename in filenames:
				match = regex.match(filename)
				if match is None:
					continue
				filename = os.path.join(dirpath, filename)
				layer, weight = match.groups()
				if layer not in result:
					result[layer] = {}
				if weight in result[layer]:
					logger.warning('Tensor weights have already been loaded '
						'for layer="%s", tensor="%s" from file: %s. We will '
						'skip this new file we just found: %s.', layer, weight,
						result[layer][weight], filename)
					continue
				result[layer][weight] = filename

		return result

	###########################################################################
	@staticmethod
	def find_compiled_layer_by_name(model, layer_name):
		""" Returns the Keras tensor associated with a given name.

			# Arguments

			model: Model instance. The Kur model. It must be compiled.
			layer_name: str. The name of the layer to find, or one of its
				aliases.

			# Return value

			The Keras tensor
		"""
		if model.compiled is None or 'raw' not in model.compiled:
			raise ValueError('The model must be compiled first.')
		if layer_name in model.output_aliases:
			target = model.output_aliases[layer_name]
		elif layer_name in model.input_aliases:
			target = model.input_aliases[layer_name]
		else:
			raise ValueError('Failed to find a layer named "{}"'
				.format(layer_name))
		for keras_layer, kur_layer in \
			zip(model.compiled['raw'].outputs, model.outputs):
			if kur_layer == target:
				return keras_layer
		raise ValueError('Did not find expected layer. This is a bug.')

	###########################################################################
	def process_loss(self, model, loss):
		""" Process the loss functions.

			# Arguments

			model: The Kur model. It must be compiled.
			loss: Loss instance, list/tuple of Loss instances, or a dictionary
				of model layer names mapped to Loss instances.
		"""
		import keras.backend as K

		if loss is None:
			num_outputs = len(model.outputs)
			logger.error('You are trying to construct a training/validation'
				'/testing model, but you haven\'t specified any loss '
				'functions. Your model has %d outputs: %s. You need to '
				'specify %d loss functions, one for each output.',
				num_outputs, ', '.join(model.outputs), num_outputs)
			raise ValueError('No loss functions were specified, but are '
				'required for training, testing, and validation.')

		if isinstance(loss, Loss):
			loss = [loss]

		if len(loss) != len(model.outputs):
			raise ValueError('Model has {} outputs, but only {} loss '
				'functions were specified.'
				.format(len(model.outputs), len(loss)))

		if isinstance(loss, (list, tuple)):
			loss = dict(zip(model.outputs, loss))

		if not isinstance(loss, (dict, OrderedDict)):
			raise ValueError('Loss functions given to "compile" should be '
				'a list/tuple, a dictionary, or a single Loss instance. '
				'Instead we received this: {} (type={})'
				.format(loss, type(loss)))

		loss_inputs = OrderedDict()
		loss_outputs = OrderedDict()
		for target, this_loss in loss.items():
			ins, out = this_loss.get_loss(
				model,
				target,
				self.find_compiled_layer_by_name(model, target)
			)
			# FIXME: Re-using a network output in different loss functions will
			# probably break, since each loss function is creating its own
			# placeholder inputs, but then we are throwing some away using
			# 'update'.
			loss_inputs.update(ins)
			loss_outputs[target] = K.mean(out)
			logger.debug('Adding additional inputs: %s',
				', '.join(x[0] for x in ins))

		total_loss = functools.reduce(
			lambda x, y: x + y,
			loss_outputs.values()
		)
		return loss_inputs, loss_outputs, total_loss

	###########################################################################
	def compile(self, model, loss=None, optimizer=None, blocking=True,
		assemble_only=False):
		""" Returns the Keras model instance.
		"""
		if model.compiled is None:
			model.compiled = {}

		if 'raw' not in model.compiled:
			import keras.models as M			# pylint: disable=import-error
			logger.debug('Instantiating a Keras model.')
			compiled = M.Model(
				input=[node.value for node in model.inputs.values()],
				output=[node.value for node in model.outputs.values()]
			)

			if logger.isEnabledFor(logging.DEBUG):
				x = io.StringIO()
				with contextlib.redirect_stdout(x):
					compiled.summary()
				for line in x.getvalue().split('\n'):
					logger.debug(line)

			model.compiled['raw'] = compiled

		else:
			logger.debug('Reusing an existing model.')
			compiled = model.compiled['raw']

		import keras.backend as K				# pylint: disable=import-error
		if loss is None and optimizer is None:
			logger.debug('Assembling an evaluation function from the model.')

			loss_inputs = loss_outputs = {}
			if not assemble_only:
				func = K.function(
					compiled.inputs + \
						[K.learning_phase()],
					compiled.outputs
				)
			key = 'evaluate'

		elif optimizer is None:
			logger.debug('Assembling a testing function from the model.')

			loss_inputs, loss_outputs, _ = \
				self.process_loss(model, loss)

			if not assemble_only:
				func = K.function(
					compiled.inputs + \
						list(loss_inputs.values()) + \
						[K.learning_phase()],
					compiled.outputs + \
						list(loss_outputs.values())
				)
			key = 'test'

		else:
			logger.debug('Assembling a training function from the model.')

			# Loss inputs: additional inputs needed by the loss function.
			# Loss outputs: output of the loss function
			loss_inputs, loss_outputs, total_loss = \
				self.process_loss(model, loss)

			updates = optimizer.get_optimizer(self)(
				compiled.trainable_weights, total_loss
			)

			if not assemble_only:
				func = K.function(
					compiled.inputs + \
						list(loss_inputs.values()) + \
						[K.learning_phase()],
					compiled.outputs + \
						list(loss_outputs.values()),
					updates=updates
				)
			key = 'train'

		logger.debug('Additional inputs for log functions: %s',
			', '.join(loss_inputs.keys()))

		input_names = compiled.input_names + \
			list(loss_inputs.keys())
		output_names = compiled.output_names + \
			list(loss_outputs.keys())

		input_shapes = [
			layer._keras_shape
			for layer in compiled.inputs
		] + [
			layer._keras_shape
			for layer in loss_inputs.values()
		]

		logger.debug('Expected input shapes: %s',
			', '.join('{}={}'.format(k, v) for k, v in \
				zip(input_names, input_shapes)
			))

		if assemble_only:
			func = None

		result = {
			'func' : func,
			'names' : {
				'input' : input_names,
				'output' : output_names
			},
			'shapes' : {
				'input' : input_shapes
			}
		}

		if logger.isEnabledFor(logging.DEBUG):
			logger.debug('Compiled model: %s', result)

		if not assemble_only:
			model.compiled[key] = result
			if blocking:
				self.wait_for_compile(model, key)

		return result

	###########################################################################
	def wait_for_compile(self, model, key):
		""" Waits for the model to finish compiling.
		"""
		if model.provider is None:
			logger.warning('No data provider available, so we cannot reliably '
				'wait for compiling to finish.')
			return

		provider = BatchProvider(
			sources=dict(zip(model.provider.keys, model.provider.sources)),
			batch_size=2,
			num_batches=1,
			randomize=False
		)
		model.supplement_provider(provider)

		weight_path = None
		tempdir = tempfile.mkdtemp()
		try:
			weight_path = os.path.join(tempdir, 'weights')
			self._save_keras(model.compiled['raw'], weight_path)

			logger.info('Waiting for model to finish compiling...')
			for batch in provider:
				self.run_batch(model, batch, key, False)

		finally:
			if weight_path and os.path.isdir(weight_path):
				try:
					self._restore_keras(model.compiled['raw'], weight_path)
				except:
					logger.error('We were waiting for the model to finish '
						'compiling, but failed to restore the model weights. '
						'The weights may be in a bad state.')
					raise

			shutil.rmtree(tempdir, ignore_errors=True)

	###########################################################################
	def run_batch(self, model, batch, key, is_train):
		if model.compiled is None or key not in model.compiled:
			raise ValueError('A model has not been compiled to: {}'
				.format(key))

		compiled = model.compiled[key]
		raw = model.compiled['raw']

		assert isinstance(is_train, bool)

		def coerce_shape(data, shape, name):
			if data.ndim < len(shape):
				return numpy.expand_dims(data, -1)
			else:
				return data

		inputs = [
			coerce_shape(
				batch[model.get_data_name_by_layer_name(batch, name)],
				shape, name
			)
			for shape, name in zip(
				compiled['shapes']['input'],
				compiled['names']['input']
			)
		] + [is_train]

		outputs = compiled['func'](inputs)
		num_outputs = len(raw.outputs)
		metrics = {
			k : v for k, v in zip(
				compiled['names']['output'][num_outputs:],
				outputs[num_outputs:]
			)
		}
		predictions = {name : data for name, data in zip(model.outputs, outputs[:num_outputs])}
		return predictions, metrics

	###########################################################################
	def train(self, model, data):
		""" Fits the given model on a batch of data.
		"""
		return self.run_batch(model, data, 'train', True)

	###########################################################################
	def test(self, model, data):
		""" Calculates the model loss on a batch of data.
		"""
		return self.run_batch(model, data, 'test', False)

	###########################################################################
	def evaluate(self, model, data):
		""" Evaluates the model on a batch of data.
		"""
		return self.run_batch(model, data, 'evaluate', False)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

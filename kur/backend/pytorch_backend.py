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
import sys
import io
import re
import logging
from collections import OrderedDict

from . import Backend
from ..utils import can_import, idx, DisableLogging
from ..loss import Loss
from ..providers import BatchProvider

logger = logging.getLogger(__name__)

###############################################################################
class PyTorchBackend(Backend):
	""" Backend which uses PyTorch.
	"""

	###########################################################################
	@classmethod
	def is_supported(cls):
		""" Returns True if this backend can be used.
		"""
		return can_import('torch')

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the backend class.

			This is used by the base class's factory method.
		"""
		return 'pytorch'

	###########################################################################
	@property
	def version(self):
		""" Returns the PyTorch version, as a tuple of (MAJOR, MINOR, PATCH).
		"""
		import torch							# pylint: disable=import-error
		version = torch.__version__
		match = re.match(r'([0-9]+)\.([0-9]+)\.([0-9]+)\.*', version)
		if not match:
			logger.warning('Unable to infer PyTorch version. We '
				'cannot check for version incompatibilities.')
			return (0, 0, 0)
		return tuple(int(x) for x in match.groups())

	###########################################################################
	def save(self, model, filename):
		""" Saves the model weights to the given filename.

			# Arguments

			model: Model instance. The model whose weights should be saved.
			filename: str. The filename to write the weights to.

			# Notes

			The file format is backend-specific. There is no guarantee of
			compatability between different backends.

			# Return value

			None
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

		for k, v in model.data.model.state_dict().items():
			layer_name, name = k.split('.', 1)
			target = os.path.join(
				path,
				'{}+{}.kur'.format(layer_name, name)
			)

			idx.save(target, v.cpu().numpy())

	###########################################################################
	def restore(self, model, filename):
		""" Load the model weights from the given filename.

			# Arguments

			model: Model instance. The model whose weights should be restored.
			filename: str. The filename to read the weights from.

			# Notes

			The file format is backend-specific. There is no guarantee of
			compatability between different backends.

			# Return value

			None
		"""

		# pylint: disable=import-error
		import torch
		# pylint: enable=import-error

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

		# Enumerate all of the saved tensors, organized like this:
		# tensors = {
		#	'layer_1_name' : {
		#		'weight_1_name'  : '/path/to/file',
		#		...,
		#	},
		#	...
		# }
		tensors = self.enumerate_saved_tensors(path)

		state = dict(model.data.model.state_dict())

		for k in state:
			layer_name, name = k.split('.', 1)
			layers = tensors.get(layer_name)
			if not layers:
				logger.warning('A model layer was not restored because there '
					'was no corresponding weight file: %s', k)
				continue
			target = layers.get(name)
			if target is None:
				logger.warning('Part of a model layer was not restored '
					'because there as no corresponding weight file: %s', k)
				continue

			state[k] = torch.from_numpy(idx.load(target))

			del layers[name]

		remaining = [v for k, V in tensors.items() for v in V]
		if remaining:
			logger.warning('Some weight files were not used when restoring '
				'the model: %s', remaining)

		model.data.model.load_state_dict(state)

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
	def create_data(self, model):
		""" Requests a new set of model-specific data to be used during model
			assembly.
		"""
		from .pytorch.modules import TorchModel
		if self.parallel:
			os.environ['CUDA_VISIBLE_DEVICES'] = \
				','.join(str(x) for x in self.devices)
			devices = tuple(range(self.parallel))
		else:
			devices = None
		data = TorchModel(gpu=devices)
		data.allow_reuse = True
		return data

	###########################################################################
	def connect(self, inputs, target, data):
		""" Applies a tensor operation to a set of input tensors.

			# Arguments

			inputs: list. A list of input tensors.
			target: object. The tensor operation to apply.

			# Notes

			The exact underlying types of the `inputs` and `target` tensors
			depend on the backend implementation.

			# Return value

			A new tensor object (backend-specific) resulting from the applied
			operation.
		"""
		if not isinstance(inputs, list):
			inputs = [inputs]
		return target(inputs)

	###########################################################################
	def process_loss(self, model, loss):
		""" Process the loss functions.

			# Arguments

			model: The Kur model. It must be compiled.
			loss: Loss instance, list/tuple of Loss instances, or a dictionary
				of model layer names mapped to Loss instances.
		"""
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

		return OrderedDict(
			(
				target,
				this_loss.get_loss(model, target, model.outputs[target].value)
			)
			for target, this_loss in loss.items()
		)

	###########################################################################
	def compile(self, model, loss=None, optimizer=None, blocking=True,
		assemble_only=False):
		""" Creates an implementation-specific representation of the
			instantiated model.

			# Arguments

			model: Model instance. The built model to create a representation
				for.
			loss: Loss instance, list, dict, or None. The loss function
				(objective function) to use for training. If None, no loss
				functions are used. If a list, it should be a list of Loss
				instances, one for each model output; if the list is length 1,
				then the same loss function is used for each model output. If
				this is a dict, it should map from model output names to Loss
				instances to apply to that output.
			optimizer: Optimizer instance or None. The loss function to use for
				training. If None, no optimizer is used.
			blocking: bool (default: True). If True, requests that the
				implementation not return until all code compiling is complete.
				If False, the implementation may spawn background threads or
				processes, and these may still be running when
				train/test/evaluate are called (leading to blocking behavior at
				that point).
			assemble_only: bool (default: False). If True, does not attempt to
				create any compiled functions/models at all. Simply assembles
				the model and prepares the data sources. If this is True, then
				blocking is ignored.

			# Return value

			An implementation-specific object ("model") that can be used for
			training or evaluation. However, if `assemble_only` is True, then
			this returns a limited object which cannot be actually used for
			anything.

			# Notes

			Implementations may need to modify the Model. If this is the case,
			ExtensionState should be used to protect the model from extensions
			that are left stuck on after compiling.
		"""
		if model.compiled is None:
			model.compiled = {}
		else:
			logger.trace('Searching for pre-compiled model.')
			if loss is None and optimizer is None:
				permissable = ('evaluate', 'test', 'train')
			elif optimizer is None:
				permissable = ('test', 'train')
			else:
				permissable = ('train', )
			new_key = permissable[0]
			for k in permissable:
				if k in model.compiled:
					logger.trace('Found pre-compiled model: %s. We can use '
						'this in place of "%s" without a problem.', k,
						new_key)
					result = model.compiled[k]
					if not assemble_only:
						model.compiled[new_key] = result
					return result

		model.data.set_outputs(
			[(k, v.value['layer']) for k, v in model.outputs.items()]
		)

		if loss is None and optimizer is None:
			key = 'evaluate'
			result = {}
		elif optimizer is None:
			torch_losses = self.process_loss(model, loss)

			key = 'test'
			result = {'loss' : torch_losses}
		else:
			trainables = tuple(model.data.get_trainable_parameters())
			if not trainables:
				torch_optimizer = None
			else:
				torch_optimizer = optimizer.get_optimizer(self)(
					trainables)

			if optimizer.clip_type:
				if self.version < (0, 1, 10):
					# Of course, we could always clip the gradients ourselves,
					# but let's just rely on PyTorch for that.
					logger.warning('Gradient clipping is only supported on '
						'PyTorch versions >= 0.1.10. We will ignore the '
						'optimizer\'s "clip" setting.')
					optimizer.clip_type = None

			torch_losses = self.process_loss(model, loss)

			key = 'train'
			result = {
				'loss' : torch_losses,
				'optimizer' : torch_optimizer,
				'kur_optimizer' : optimizer
			}

		result.update({
			'model' : model.data
		})

		if logger.isEnabledFor(logging.DEBUG):
			x = io.StringIO()
			self.summary(model, x)
			for line in x.getvalue().split('\n'):
				logger.debug(line)

		if not assemble_only:
			model.compiled[key] = result
			if blocking:
				self.wait_for_compile(model, key)

		return result

	###########################################################################
	def wait_for_compile(self, model, key):
		""" Waits for the PyTorch model to be ready to roll.

			# Notes:

			PyTorch models are not compiled in the same way Theano/TensorFlow
			models are prepared. However, if you are using the GPU, then there
			is still overhead associated with spinning up CUDA. Theano and
			TensorFlow do this shortly after being initialized, but PyTorch
			waits until the last necessary minute. So although this call might
			more accurately be called `wait_for_cuda_devices`, we keep the name
			`wait_for_compile` to keep the API similar to, e.g., the Keras
			backend.
		"""
		if model.provider is None:
			logger.warning('No data provider available, so we cannot reliably '
				'wait for compiling to finish.')
			return

		with DisableLogging():
			provider = BatchProvider(
				sources=dict(zip(model.provider.keys, model.provider.sources)),
				batch_size=2*max(1, self.parallel),
				num_batches=1,
				randomize=False
			)
		model.supplement_provider(provider)

		logger.info('Waiting for model to be ready to use...')
		for batch in provider:
			model.data.predict(batch)
		logger.info('Model is ready for use.')

	###########################################################################
	def summary(self, model, file=None):
		""" Prints a model summary
		"""
		file = file or sys.stdout
		fill = lambda x, w: x[:min(w, len(x))].ljust(w)
		num_param = 0
		print('{}-+-{}-+-{}'.format('-'*30, '-'*20, '-'*10), file=file)
		print('{} | {} | {}'.format(
			fill('Layer Name', 30),
			fill('Shape', 20),
			fill('Parameters', 10)
		), file=file)
		print('{}-+-{}-+-{}'.format('-'*30, '-'*20, '-'*10), file=file)
		for k, v in model.data.model.state_dict().items():
			print('{} | {} | {}'.format(
				fill(k, 30),
				fill(str(tuple(v.size())), 20),
				fill(str(v.numel()), 10)
			), file=file)
			print('{}-+-{}-+-{}'.format('-'*30, '-'*20, '-'*10), file=file)
			num_param += v.numel()
		print('Total parameters: {}'.format(num_param), file=file)

	###########################################################################
	def train(self, model, data):
		""" Fits the given model on a batch of data.

			# Arguments

			model: Model instance. The model to train with.
			data: dict or list. The data to train a batch on.
			compiled: object. The data package returned by `compile`. Its type
				and content are backend-specific.

			# Return value

			A dictionary of loss values, whose keys are the names of the
			respective outputs. If per-output loss is not available, then a
			single, global loss value can be returned instead.
		"""

		torch_model = model.compiled['train']['model']
		losses = model.compiled['train']['loss']
		optimizer = model.compiled['train']['optimizer']
		kur_optimizer = model.compiled['train']['kur_optimizer']

		if optimizer:
			optimizer.zero_grad()

		predictions, losses = torch_model.test(data, losses)

		if optimizer:
			torch_model.backprop(losses)

			if kur_optimizer.clip_type:
				torch_model.clip_gradients(
					kur_optimizer.clip_type,
					kur_optimizer.clip_value
				)

			step_done = False
			if kur_optimizer.scale_rate:
				if kur_optimizer.scale_rate in data:
					factor = data[kur_optimizer.scale_rate].mean()
					for param_group in optimizer.param_groups:
						param_group['lr'] *= factor
					optimizer.step()
					for param_group in optimizer.param_groups:
						param_group['lr'] /= factor
					step_done = True
				else:
					logger.warning('The optimizer "scale_rate" was specified, '
						'but no such data column was found: %s. Ignoring '
						'this.', kur_optimizer.scale_rate)
			if not step_done:
				optimizer.step()

		metrics = {
			k : loss.data.cpu().numpy().squeeze(-1)
			for k, loss in zip(model.outputs, losses)
		}

		predictions = {
			k : v.data.cpu().numpy() for k, v in zip(model.outputs, predictions)
		}

		return (predictions, metrics)

	###########################################################################
	def test(self, model, data):
		""" Calculates the model loss on a batch of data.

			# Arguments

			model: Model instance. The model to evaluate with.
			data: dict or list. The batch of data to test on.
			compiled: object. The data package returned by `compile`. Its type
				and content are backend-specific.

			# Return value

			A dictionary of loss values, whose keys are the names of the
			respective outputs. If per-output loss is not available, then a
			single, global loss value can be returned instead.
		"""
		torch_model = model.compiled['test']['model']
		losses = model.compiled['test']['loss']

		predictions, losses = torch_model.test(data, losses)

		metrics = {
			k : loss.data.cpu().numpy().squeeze(-1)
			for k, loss in zip(model.outputs, losses)
		}

		predictions = {
			k : v.data.cpu().numpy() for k, v in zip(model.outputs, predictions)
		}

		return (predictions, metrics)

	###########################################################################
	def evaluate(self, model, data):
		""" Evaluates the model on a batch ofdata.
		"""
		torch_model = model.compiled['evaluate']['model']

		predictions = torch_model.predict(data)

		predictions = {
			k : v.data.cpu().numpy() for k, v in zip(model.outputs, predictions)
		}

		return (predictions, {})

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

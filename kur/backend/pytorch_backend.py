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
from ..utils import can_import, idx
from ..loss import Loss

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
		data = TorchModel(gpu=self.device == 'gpu')
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
			torch_optimizer = optimizer.get_optimizer(self)(
				model.data.model.parameters())
			torch_losses = self.process_loss(model, loss)

			key = 'train'
			result = {'loss' : torch_losses, 'optimizer' : torch_optimizer}

		result.update({
			'model' : model.data
		})

		if not assemble_only:
			model.compiled[key] = result

		if logger.isEnabledFor(logging.DEBUG):
			x = io.StringIO()
			self.summary(model, x)
			for line in x.getvalue().split('\n'):
				logger.debug(line)

		return result

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
		optimizer = model.compiled['train']['optimizer']
		losses = model.compiled['train']['loss']

		optimizer.zero_grad()

		predictions, losses = torch_model.test(data, losses)

		torch_model.backprop(losses)

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

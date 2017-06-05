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

import re
import logging
from collections import OrderedDict

# pylint: disable=import-error
import torch
import torch.nn as nn
from torch.autograd import Variable
# pylint: enable=import-error

import numpy

logger = logging.getLogger(__name__)

###############################################################################
class BYOM(nn.Module):				# pylint: disable=too-few-public-methods
	""" Bring-Your-Own-Module for PyTorch.

		This is just a simple PyTorch module that allows you to build up your
		module in pieces (using `setattr` to register parameters, and `func` to
		perform the forward pass). It gives you the benefits of a Module with
		the flexibility of Kur's graph interpretation.
	"""

	###########################################################################
	def __init__(self):
		""" Creates a new Module.
		"""
		super().__init__()
		self.func = None

	###########################################################################
	def forward(self, *inputs):
		""" Performs the forward pass.
		"""
		assert self.func is not None
		return self.func(self, *inputs)

###############################################################################
class Layer:
	""" Holds a PyTorch "layer". This is important because PyTorch copies the
		Modules for each GPU; that is, you cannot hold on to Module refernces,
		or even layer references, since they may change. This Layer class
		abstracts that away by dynamically grabbing the correct layer instance
		during the forward pass.
	"""

	###########################################################################
	def __init__(self, name, func=None):
		""" Creates a new layer.
		"""
		self.name = name
		self.func = func

	###########################################################################
	def __call__(self, module, *x):
		""" Grab the instantiated layer and evaluate it.
		"""
		operation = getattr(module, self.name)
		if self.func:
			return self.func(operation, *x)
		return operation(*x)

	###########################################################################
	@staticmethod
	def resolve(value):
		""" Convenience function for calling either Layers or standard PyTorch
			operations.
		"""
		if isinstance(value, Layer):
			return value
		elif hasattr(value, 'pure') and value.pure:
			return value
		return lambda _, *args: value(*args)

###############################################################################
class TorchModel:
	""" This holds the Torch graph and provides convenience functions for using
		it.
	"""

	DATA_CAST = {
		'int' : lambda x: x.int(),
		'long' : lambda x: x.long(),
		'float' : lambda x: x.float(),
		'double' : lambda x: x.double()
	}

	###########################################################################
	def __init__(self, gpu=None):
		""" Creates a new model.
		"""
		self.model = BYOM()
		self.inputs = []
		self.outputs = None
		self.layer_map = {}
		self.gpu = gpu
		self._reuse = False
		self.final_model = None
		self.info = []

	###########################################################################
	@property
	def allow_reuse(self):
		""" Getter for the `allow_reuse` property, which, if enabled, causes
			`add_layer` calls to reuse existing layer when possible, rather
			than raising an exception about duplicate names.
		"""
		return self._reuse

	###########################################################################
	@allow_reuse.setter
	def allow_reuse(self, value):
		self._reuse = value

	###########################################################################
	def set_outputs(self, outputs):
		""" Sets the model outputs.

			# Arguments

			outputs: an OrderedDict, or a list of `(name, layer)` tuples. Each
				`name` (or key of the OrderedDict) is the name of a layer, and
				each `layer` is a function as returned by `add_operation` or
				`add_layer`, once that function has been applied to other
				layers.
		"""
		self.outputs = OrderedDict(outputs)
		self.model.func = self.add_operation(bundle)(
			*self.outputs.values()
		)
		self.final_model = self.parallelize()

	###########################################################################
	def parallelize(self):
		""" Applies any parallelism requested.
		"""
		if not self.gpu:
			return self.model
		if isinstance(self.gpu, bool):
			devices = None
		else:
			devices = self.gpu
		return nn.DataParallel(self.model, devices).cuda()

	###########################################################################
	def to_torch(self, tensor, *, location=None, data_type=None):
		""" Creates a Torch tensor from an array.

			# Arguments

			tensor: one of tuple, list, numpy.ndarray, torch.Tensor
		"""
		#if self.gpu:
		#	if numpy_tensor.dtype.kind == 'f' and \
		#			numpy_tensor.dtype.itemsize != 4:
		#		numpy_tensor = numpy_tensor.astype(
		#			'{}f4'.format(numpy_tensor.dtype.byteorder)
		#		)
		if isinstance(tensor, (list, tuple)):
			tensor = numpy.array(tensor)
		if isinstance(tensor, numpy.ndarray):
			tensor = torch.from_numpy(tensor)
		tensor = self.DATA_CAST.get(data_type or 'float')(tensor)
		if self.gpu:
			if location != 'cpu':
				tensor = tensor.cuda()
		return tensor

	###########################################################################
	def predict(self, data):
		""" Performs the forward pass.
		"""
		inputs = tuple(
			Variable(self.to_torch(
				data[k], location=info['location'], data_type=info['type']
			))
			for k, info in zip(self.inputs, self.info)
		)
		return self.final_model(*inputs)

	###########################################################################
	def cpu(self, x):
		""" Moves a tensor (or list/tuple of tensors) back to the CPU.
		"""
		if isinstance(x, (list, tuple)):
			return tuple(X.cpu() for X in x)
		return x.cpu()

	###########################################################################
	def test(self, data, loss):
		""" Performs a forward pass and calculates loss.

			# Arguments

			data: dict. A dictionary whose keys are the names of the model
				layers, and whose respective values are numpy arrays contain
				those values for the batch. For this function, both model
				inputs and outputs must be specified.
			loss: a list of loss functions, each one corresponding to the
				respective model output. These loss functions should take two
				arguments; the first is a list of the tensors needed to compute
				loss (e.g., the ground truth data), and the second is the model
				output to compare against.
		"""
		inputs = tuple(
			Variable(self.to_torch(
				data[k], location=info['location'], data_type=info['type']
			))
			for k, info in zip(self.inputs, self.info)
		)
		predictions = self.final_model(*inputs)

		#######################################################################
		def get_loss(loss, prediction):
			""" Calculates the loss given a loss specification.
			"""
			loss_inputs = [x[1](None, *inputs) for x in loss[0]]
			return loss[1](loss_inputs, prediction)

		losses = [
			get_loss(loss[output], P)
			for output, P in zip(self.outputs, predictions) if output in loss
		]

		return predictions, losses

	###########################################################################
	def move(self, module):
		""" Moves a module to the GPU, if the GPU is enabled.
		"""
		if self.gpu:
			return module.cuda()
		return module

	###########################################################################
	def placeholder(self, name, *, create=True, location=None, data_type=None):
		""" Creates a new input placeholder, or retrieves an existing one.
		"""

		# Placeholders are just named ways to access one of the input tensors.
		try:
			index = self.inputs.index(name)
		except ValueError:
			if not create:
				return None
			index = len(self.inputs)
			self.inputs.append(name)
			self.info.append({
				'location' : location,
				'type' : data_type
			})

		#######################################################################
		def calculate(_, *inputs):
			""" Applies the layer.
			"""
			if index >= len(inputs):
				raise IndexError('Out-of-range index: {}. Must be < {}. Note: '
					'we are trying to find "{}".'
					.format(index, len(inputs), calculate.target))
			return inputs[index]
		calculate.target = name
		calculate.index = index
		calculate.name = name

		return calculate

	###########################################################################
	@staticmethod
	def normalize_name(name):
		""" Creates a PyTorch-compatible layer name.

			In PyTorch, "layers" can be automatically tracked by a Module, but
			only if they are class attributes. So we will take a name, and get
			a Pythonically-allowed variable to use for the attribute.
		"""
		new_name = re.sub(r'^[^a-zA-Z_]|[^a-zA-Z0-9_]', r'_', name)
		new_name = 'layer_{}'.format(new_name)
		return new_name

	###########################################################################
	def backprop(self, losses):
		""" Runs the backward pass on the network.
		"""
		grads = [self.to_torch([1.0]) for _ in range(len(losses))]
		torch.autograd.backward(
			losses,
			grads
		)

	###########################################################################
	def clip_gradients(self, clip_type, clip_value):
		if clip_type == 'norm':
			norm_type = 2
		elif clip_type == 'abs':
			norm_type = 'inf'
		else:
			raise ValueError('Clip type must be "norm" or "abs".')

		nn.utils.clip_grad_norm(
			self.get_trainable_parameters(),
			clip_value,
			norm_type
		)

	###########################################################################
	def add_operation(self, operation, name=None):# pylint: disable=no-self-use
		""" Adds a new operation to the graph.

			# Notes

			- Operations have no learnable parameters, and have no names.
		"""

		if name is None:
			name = operation.__name__

		#######################################################################
		def stack(*lower_layers):
			""" Returns a function suitable for using in a functional paradigm
				of model assemble.
			"""

			logger.trace('Connecting layers: %s feed into %s',
				[
					x.name if hasattr(x, 'name') else 'unknown'
					for x in lower_layers
				], name
			)

			###################################################################
			def calculate(module, *inputs):
				""" Applies the layer.
				"""
				result = Layer.resolve(operation)(
					module,
					*[x(module, *inputs) for x in lower_layers]
				)
				return result
			calculate.name = name
			calculate.op = operation

			return calculate

		stack.name = name

		return stack

	###########################################################################
	def add_variable(self, name, value, func=None):
		""" Adds a new variable.
		"""
		new_name = self.normalize_name(name)
		if name in self.layer_map:
			if not self.allow_reuse:
				raise ValueError('Duplicate name found: {}'.format(name))
		else:
			self.layer_map[name] = new_name
			logger.trace('Adding new layer: %s (internal: "%s")',
				name, new_name)
			setattr(self.model, new_name, value)

		return Layer(new_name, func)

	###########################################################################
	def add_layer(self, name, layer, *, func=None, frozen=False):
		""" Creates a new layer.

			# Notes

			- The layer must not already exist.
			- The layer will be registered as potentially containing learnable
			  parameters.
			- All learnable layers must be added using this function.
		"""
		if frozen is not None:
			for param in layer.parameters():
				param.requires_grad = not frozen
		layer = self.add_variable(name, layer, func)
		return self.add_operation(layer, name=name)

	###########################################################################
	def get_trainable_parameters(self):
		""" Returns a generator over all trainable model parameters.
		"""
		for param in self.model.parameters():
			if param.requires_grad:
				yield param

###############################################################################
def flatten(x):
	""" Flattens an input tensor.
	"""
	return x.view(x.size()[0], -1)

###############################################################################
def bundle(*x):
	""" Merge tensors.
	"""
	return x

###############################################################################
def move_channel_forward(x):
	ndim = x.dim()
	permutation = (0, ndim-1) + tuple(range(1, ndim-1))
	return x.permute(*permutation)

###############################################################################
def move_channel_backward(x):
	ndim = x.dim()
	permutation = (0, ) + tuple(range(2, ndim)) + (1, )
	return x.permute(*permutation)

###############################################################################
class swap_channels:
	""" Swaps the dimensions between Theano/PyTorch and TensorFlow dimension
		orderings.
	"""
	begin = move_channel_forward
	end = move_channel_backward

###############################################################################
def parallel(layer):
	""" Creates a parallel operation (i.e., map/distributed operation).
	"""
	def func(module, x):
		""" The actual wrapped operation.
		"""
		return torch.stack(
			tuple(Layer.resolve(layer)(module, X) for X in torch.unbind(x, 0)),
			0
		)
	func.pure = True
	return func

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

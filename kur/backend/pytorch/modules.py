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
from functools import partial

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
		return self.func(*inputs)

###############################################################################
class TorchModel:
	""" This holds the Torch graph and provides convenience functions for using
		it.
	"""

	###########################################################################
	def __init__(self, gpu=False):
		""" Creates a new model.
		"""
		self.model = BYOM()
		self.inputs = []
		self.outputs = None
		self.layer_map = {}
		self.gpu = gpu

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
		if self.gpu:
			self.model.cuda()

	###########################################################################
	def to_torch(self, tensor):
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
		tensor = tensor.float()
		if self.gpu:
			tensor = tensor.cuda()
		return tensor

	###########################################################################
	def predict(self, data):
		""" Performs the forward pass.
		"""
		inputs = tuple(
			Variable(self.to_torch(data[k]))
			for k in self.inputs
		)
		return self.model(*inputs)

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
			Variable(self.to_torch(data[k])) \
			for k in self.inputs
		)
		predictions = self.model(*inputs)

		#######################################################################
		def get_loss(loss, prediction):
			""" Calculates the loss given a loss specification.
			"""
			loss_inputs = [x[1](*inputs) for x in loss[0]]
			return loss[1](loss_inputs, prediction)

		losses = [
			get_loss(loss[output], P)
			for output, P in zip(self.outputs, predictions)
		]

		return predictions, losses

	###########################################################################
	def placeholder(self, name, create=True):
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

		#######################################################################
		def calculate(*inputs):
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
	def get_layer(self, name):
		""" Retrieves a layer in the model by name.

			# Return value

			The PyTorch module with that name, or None if it does not exist.
		"""
		new_name = self.normalize_name(name)
		result = self.layer_map.get(new_name)
		if result is None:
			return self.placeholder(name, create=False)
		return result

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

			logger.debug('Connecting layers: %s feed into %s',
				[x.name for x in lower_layers], name
			)

			###################################################################
			def calculate(*inputs):
				""" Applies the layer.
				"""
				result = operation(*[x(*inputs) for x in lower_layers])
				return result
			calculate.name = name

			return calculate
		stack.name = name

		return stack

	###########################################################################
	def add_variable(self, name, value):
		""" Adds a new variable.
		"""
		if name in self.layer_map:
			raise ValueError('Duplicate name found: {}'.format(name))
		new_name = self.normalize_name(name)
		self.layer_map[name] = new_name

		setattr(self.model, new_name, value)
		return value

	###########################################################################
	def add_layer(self, name, layer, func=None):
		""" Creates a new layer.

			# Notes

			- The layer must not already exist.
			- The layer will be registered as potentially containing learnable
			  parameters.
			- All learnable layers must be added using this function.
		"""
		self.add_variable(name, layer)

		if func is None:
			func = layer
		else:
			func = partial(func, self, layer)

		return self.add_operation(func, name=name)

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
def swap_channels(x):
	""" Swaps the dimensions between Theano/PyTorch and TensorFlow dimension
		orderings.
	"""
	return torch.transpose(x, 1, x.dim()-1)

###############################################################################
def parallel(layer):
	""" Creates a parallel operation (i.e., map/distributed operation).
	"""
	def func(*x):
		""" The actual wrapped operation.
		"""
		return torch.cat(tuple(layer(X) for X in torch.unbind(x, 0)), 0)
	return func

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

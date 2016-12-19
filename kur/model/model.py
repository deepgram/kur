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
import types
from collections import namedtuple, OrderedDict, deque
from ..containers import Container
from ..containers.operators import ContainerGroup
from ..containers.layers import Placeholder
from ..engine import PassthroughEngine

logger = logging.getLogger(__name__)

# Convenience class for keeping track of high-level network nodes.
ContainerNode = types.SimpleNamespace

CollapsedContainer = namedtuple('CollapsedContainer',
	['inputs', 'container', 'names']
)

###############################################################################
class Node:							# pylint: disable=too-few-public-methods
	""" Class for representing nodes in the low-level neural network graph.

		Each Node instance represents one primitive tensor operation.
	"""

	# Counter for tracking node IDs (used for debugging purposes).
	counter = 0

	###########################################################################
	def __init__(self, inputs=None, operation=None, value=None, outputs=None):
		""" Creates a new Node.

			# Arguments

			inputs: list of Nodes. The Nodes that are inputs to this operation.
			operation: object. The tensor operation to perform; the type is
				backend-specific.
			value: object. The output of the tensor operation, or None to
				indicate that it hasn't been evaluated yet.
			outputs: list of Nodes. The Nodes that use this node as input.
		"""
		self.inputs = inputs
		self.operation = operation
		self.value = value
		self.outputs = outputs

		# Increment and store the Node ID counter for debugging purposes.
		self.node_id = Node.counter
		Node.counter += 1

	###########################################################################
	def __str__(self):
		""" Return a string representation of this Node, suitable for
			debugging.
		"""
		return 'Node(id={node_id}, inputs={inputs}, operation={operation}, ' \
			'value={value}, outputs={outputs})'.format(
				node_id=self.node_id,
				inputs=[node.node_id for node in self.inputs]
					if self.inputs else None,
				operation=self.operation,
				value=self.value,
				outputs=[node.node_id for node in self.outputs]
					if self.outputs else None
			)

###############################################################################
class ExtensionCallback:
	""" Callback interface for Model's extension callback hooks.
	"""
	###########################################################################
	def extended(self, name):
		""" Callback after an extension is added to a model.

			# Arguments

			name: str. The name of the extension that was added.
		"""
		raise NotImplementedError

	###########################################################################
	def retracted(self, name):
		""" Callback after an extension is removed from a model.

			# Arguments

			name: str. The name of the extension that was removed.
		"""
		raise NotImplementedError

###############################################################################
class ExtensionState(ExtensionCallback):\
	# pylint: disable=too-few-public-methods
	""" Context management for saving the state of the model's extensions.

		This is useful for allowing an arbitrary number of temporary changes to
		the model that should be discarded when the context is exited.
	"""

	###########################################################################
	def __init__(self, model):
		""" Create a new ExtensionState.

			# Arguments

			model: Model instance. The model to track the extension state of.
		"""
		self.model = model
		self.changed = False
		self.extensions = OrderedDict()

	###########################################################################
	def __enter__(self):
		""" Enter context management and keep track of the extensions that
			change.
		"""
		logger.debug('Entering a model state context.')
		self.extensions = OrderedDict(self.model.extensions.items())
		self.changed = False
		self.model.add_extension_callback(self)
		return self

	###########################################################################
	def __exit__(self, exc_type, exc_value, traceback):
		""" Exit context management and restore the extension state.
		"""
		logger.debug('Leaving a model state context.')
		self.model.remove_extension_callback(self)

		if self.changed:
			logger.debug('Reverting changes during extension state '
				'monitoring.')

			# Remove all current extensions.
			for name in reversed(self.model.extensions):
				self.model.retract(name, rebuild=False)

			# Add all the previous extensions.
			for name, containers in self.extensions.items():
				self.model.extend(name, containers, rebuild=False)

			# Build the model.
			self.model.build()
		else:
			logger.debug('Nothing changed during extension state monitoring.')

	###########################################################################
	def extended(self, name):
		""" Callback after an extension is added to a model.
		"""
		self.changed = True

	###########################################################################
	def retracted(self, name):
		""" Callback after an extension is removed from a model.
		"""
		self.changed = True

###############################################################################
class Extension:					# pylint: disable=too-few-public-methods
	""" Context management for model extension.

		This is used to temporarily add an extension to a model. The extension
		is added when the context is entered, and removed when the context is
		exited.
	"""

	###########################################################################
	@staticmethod
	def _token(x):
		""" Produces random strings of hexadecimal digits.

			If the 'secrets' module is present (Python 3.6+), it is used.
			Otherwise, 'random' is used.

			# Arguments

			x: int. The number of bytes to generate.

			# Return value

			A string of length 2*n.
		"""
		try:
			import secrets
			return secrets.token_hex(x)
		except ImportError:
			import hashlib
			import random
			state = hashlib.sha256()
			result = ''
			while len(result) < x*2:
				state.update(bytes(random.randint(0, 255) for i in range(100)))
				result += state.hexdigest()
			return result[:x*2]

	###########################################################################
	def __init__(self, model, containers, name=None):
		""" Creates a new extension.

			# Arguments

			model: Model instance. The model to modify.
			containers: list of Containers. The containers to extend the model
				with.
			name: str (default: None). The name of the extension. If None, a
				random name is used.
		"""
		self.model = model
		self.containers = containers
		self.name = name or Extension._token(16)

	###########################################################################
	def __enter__(self):
		""" Enter the context manager and apply the extension.
		"""
		self.model.extend(self.name, self.containers)
		return self

	###########################################################################
	def __exit__(self, exc_type, exc_value, traceback):
		""" Exit the context manager and remove the extension.
		"""
		self.model.retract(self.name)

###############################################################################
class Model:
	""" Class for holding the model corresponding to a given container.

		Unlike a container, a model's layers are connected vis-a-vis the
		backend, and it has tracked input and output connections.
	"""

	###########################################################################
	def __init__(self, backend, containers):
		""" Create a new model.

			# Arguments

			backend: Backend instance. The backend to use in assembling this
				model.
			container: Container or list of Containers. The container(s) to
				assemble in a model. If this is a list, then note that order is
				important, since layers without explicit 'input' connections
				will be connected to the previous layer.
		"""
		if isinstance(containers, Container):
			containers = [containers]
		self.root = ContainerGroup(containers)

		self.backend = backend

		self.extensions = OrderedDict()
		self.extension_callbacks = []

		self.network = None
		self.inputs = None
		self.outputs = None
		self.endpoints = None
		self.container_nodes = None

		self._parsed = False

		self.input_aliases = {}
		self.output_aliases = {}
		self.key_cache = {}

		self.provider = None
		self.additional_sources = {}

	###########################################################################
	def has_data_source(self, name):
		""" Checks if an auxiliary data source is present.
		"""
		return name in self.additional_sources

	###########################################################################
	def add_data_source(self, name, source):
		""" Registers an auxiliary data source.
		"""
		self.additional_sources[name] = source

	###########################################################################
	def get_data_sources(self):
		""" Returns an iterator over (key, value) tuples of auxiliary data
			sources.
		"""
		return self.additional_sources.items()

	###########################################################################
	def register_provider(self, provider):
		""" Let's the model know which data provider we are using.

			This allows the model to do shape inference for layers.
		"""
		self.provider = provider

	###########################################################################
	def get_inferred_shape(self, name):
		""" Tries to infer the shape of a container from the data.

			# Arguments

			name: str. The name of the data source.

			# Return value

			If a data provider has been registered (through a previous call to
			`register_provider()`, and if `name` is one of that provider's data
			sources, then the shape of that source is returned (as a tuple).
			Otherwise, None is returned.
		"""
		logger.debug('Trying to infer shape for input "%s"', name)
		if self.provider is None:
			logger.debug(
				'No provider has been registered to use for shape inference.')
			return None
		if self.provider.keys is None:
			logger.debug(
				'The provider does not have data keys associated with it.')
			return None

		try:
			index = self.provider.keys.index(name)
		except ValueError:
			logger.warning(
				'No such data source found for shape inference: %s', name)
			return None

		shape = self.provider.sources[index].shape()
		logger.debug('Inferred shape for input "%s": %s', name, shape)
		return shape

	###########################################################################
	def get_data_name_by_layer_name(self, keys, layer_name):
		""" Finds the key of a data dictionary which is intended to provide
			data to a particular layer in the model.

			# Example

			```python
			# Say that you have data of this form:
			data = {'Aaron' : _, 'Barbara' : _, 'Charles' : _}
			# And the set of model aliases is:
			input_aliases = {'Apple' : 'A', 'Aaron' : 'A', 'Beth' : 'B', ...}
			# Then:
			assert get_data_name_by_layer_name(
				data.keys(),
				'A'
			) == 'Aaron'
			```
		"""
		key = self.key_cache.get(layer_name)
		if key in keys:
			return key

		for alias_list in (self.input_aliases, self.output_aliases):
			for key, name in alias_list.items():
				if key in keys and name == layer_name:
					self.key_cache[layer_name] = key
					return key

		logger.error('No such layer name found: "%s". Something will probably '
			'break in a moment. This is probably a bug.', layer_name)
		return layer_name

	###########################################################################
	def get_layer_name_by_data_name(self, data_name):
		""" Finds the canonical name of the layer which uses a particular data
			source.
		"""
		if data_name in self.input_aliases:
			return self.input_aliases[data_name]
		if data_name in self.output_aliases:
			return self.output_aliases[data_name]
		logger.error('No such data name found: "%s". Something will probably '
			'break in a moment. This is probably a bug.', data_name)
		return data_name

	###########################################################################
	def get_backend(self):
		""" Returns the backend this model is using.
		"""
		return self.backend

	###########################################################################
	def add_extension_callback(self, extension_callback):
		""" Adds an extension callback.

			Extension callbacks are called just after a new extension is added
			and immediately before an existing extension is removed.

			# Arguments

			extension_callback: ExtensionCallback instance. The callback to
				register.
		"""
		self.extension_callbacks.append(extension_callback)

	###########################################################################
	def remove_extension_callback(self, extension_callback):
		""" Removes an extension callback.

			# Arguments

			extension_callback: ExtensionCallback instance. The callback to
				register.
		"""
		self.extension_callbacks.remove(extension_callback)

	###########################################################################
	def save(self, filename):
		""" Saves the model weights to the given filename.

			# Arguments

			filename: str. The filename to write the weights to.

			# Notes

			The file format is backend-specific. There is no guarantee of
			compatability between different backends.
		"""
		logger.debug('Saving model weights to: %s', filename)
		self.backend.save(self, filename)

	###########################################################################
	def restore(self, filename):
		""" Load the model weights from the given filename.

			# Arguments

			filename: str. The filename to read the weights from.

			# Notes

			The file format is backend-specific. There is no guarantee of
			compatability between different backends.
		"""
		logger.debug('Loading model weights from: %s', filename)
		self.backend.restore(self, filename)

	###########################################################################
	def parse(self, engine):
		""" Parses the model.
		"""
		if not self._parsed:
			if engine is None:
				logger.info('Creating a dummy engine for parsing the model.')
				engine = PassthroughEngine()
			self.root.parse(engine)
			self._parsed = True

	###########################################################################
	def is_built(self):
		""" Returns True if this model has been built at some point.
		"""
		return self.network is not None

	###########################################################################
	def build(self):
		""" Builds the model.
		"""

		if not self._parsed:
			logger.warning('The model has not been parsed yet. We will try to '
				'parse it without context, but this may easily fail. Make '
				'sure Model.parse() is called before Model.build().')
			self.parse(None)

		logger.info('Enumerating the model containers.')

		# Construct the high-level network nodes.
		nodes = self.enumerate_nodes(self.root)

		logger.info('Assembling the model dependency graph.')
		input_nodes, output_nodes, network = self.assemble_graph(nodes)

		if logger.isEnabledFor(logging.DEBUG):
			queue = deque(input_nodes.values())
			while queue:
				node = queue.popleft()
				logger.debug('Assembled Node: %s', node.container.name)
				logger.debug('  Uses: %s', ', '
					.join([x.container.name for x in node.inputs]))
				logger.debug('  Used by: %s', ', '
					.join([x.container.name for x in node.outputs]))
				logger.debug('  Aliases: %s', ', '.join(node.names))
				queue.extend(node.outputs)

		logger.info('Connecting the model graph.')
		inputs, input_aliases, outputs, output_aliases = \
			self.build_graph(input_nodes, output_nodes, network)

		logger.info('Model inputs:  %s', ', '.join(node for node in inputs))
		logger.info('Model outputs: %s', ', '.join(node for node in outputs))

		self.inputs = inputs
		self.outputs = outputs
		self.network = network
		self.input_aliases = input_aliases
		self.output_aliases = output_aliases
		self.key_cache = {}

		#assert False

	###########################################################################
	def build_graph(self, input_nodes, output_nodes, network):
		""" Builds and connects the model's underlying tensor operations.
		"""
		from ..utils.flatiter import flatten

		# 1-to-1 mapping: canonical name to compiled object.
		inputs = OrderedDict()
		# Many-to-one map: every name the input might have should map to the
		# key used in `inputs`.
		input_aliases = {}
		outputs = OrderedDict()
		output_aliases = {}

		for node in network.values():
			logger.debug('Building node: %s', node.container.name)
			logger.debug('  Aliases: %s', ', '.join(node.names))
			logger.debug('  Inputs:')
			for x in node.inputs:
				logger.debug('  - %s: %s', x.container.name, x.value)

			if node.inputs:

				# If you say that node N gets input from node M, you expect a
				# single tensor to be produced by M. After all, N is also going
				# to produce a single tensor. So you can always expect previous
				# nodes to produce single tensors.
				# If the node M's value is None, it hasn't modified the inputs
				# in any way. But if it only can produce a single output, then
				# not modifying the inputs implies that is also has only one
				# input.

				value = list(flatten([
					x.value for x in node.inputs
				]))

				for layer in node.container.build(self):
					if layer is None:
						continue
					value = self.backend.connect(
						inputs=value,
						target=layer
					)
				node.value = value

			else:
				value = None
				for layer in node.container.build(self):
					if layer is None:
						continue
					if value is None:
						value = layer
						# Register inputs
						inputs[node.container.name] = node #value
						for name in node.names:
							input_aliases[name] = node.container.name
					else:
						value = self.backend.connect(
							inputs=value,
							target=layer
						)
				# Comments:
				# - If your layer didn't produce anything, then `value` is
				#   None, and the `inputs` for this layer is set to None.
				# - If you decide to have your first layers produce None, then
				#   presumably you can figure out what to do with that value
				#   in upstream calls to `Backend.connect()`.
				# - If this layer produced nothing at all (not simply a None
				#   layer, but truly no layers at all), then `value` is still
				#   None

				node.value = value

			logger.debug('  Value: %s', node.value)

			# Register outputs
			if node.container.name in output_nodes:
				outputs[node.container.name] = node #.value
				for name in node.names:
					output_aliases[name] = node.container.name

		return inputs, input_aliases, outputs, output_aliases

	###########################################################################
	def assemble_graph(self, nodes):
		""" Creates the dependency graph of containers in the model.
		"""

		name_map = {name : node.container
			for node in nodes for name in node.names}

		# Instantiate the network.
		# Map names to resolved nodes.
		network = OrderedDict()
		for node in nodes:
			network[node.container.name] = ContainerNode(
				inputs=[],
				container=node.container,
				outputs=[],
				names=node.names,
				value=None
			)

		# Populate the inputs.
		recent = deque()
		for node in nodes:
			# Get the new node.
			resolved = network[node.container.name]

			# Check if it has inputs.
			if node.inputs:
				if isinstance(node.container, Placeholder):
					raise ValueError('Input nodes cannot themselves have '
						'inputs. It looks like container "{}" tried to, '
						'though.'.format(node.container.name))
				# It has explicit inputs. Use them.
				for name in node.inputs:
					# This name is not necessarily the container's canonical
					# name. Let's find it.
					if name in name_map:
						# Find the container by a name.
						container = name_map[name]
						# Get the ContainerNode instance by canonical name.
						input_node = network[container.name]
						# Add the input.
						resolved.inputs.append(input_node)
					else:
						# Maybe this is supposed to be an input?
						raise ValueError('Container "{}" is trying to attach '
							'to input "{}", but we cannot find that input.'
							.format(node.container.name, name))
			else:
				# It does not have explicit inputs.
				if isinstance(node.container, Placeholder):
					# No worries, it's just a new input.
					recent = deque()
				elif recent:
					# There is something to connect to behind this node.
					resolved.inputs.append(recent[-1])
				else:
					# There is nothing there.
					# This should be an input layer.
					raise ValueError('Container "{}" looks like it is '
						'supposed to be an input layer. If that is true, '
						'then it ought to be marked as "{}".'
						.format(
							node.container.name,
							Placeholder.get_container_name()
						))

			recent.append(resolved)

		# Populate the outputs.
		for node in network.values():
			for input_node in node.inputs:
				input_node.outputs.append(node)

		used_as_output = set()

		input_names = set()
		output_names = set()
		for name, node in network.items():
			if not node.inputs:
				input_names.add(name)
			else:
				# Mark its inputs as not outputs.
				for input_node in node.inputs:
					used_as_output.add(input_node.container.name)

			if node.container.sink:
				output_names.add(name)

		for name in set(network.keys()) - used_as_output:
			output_names.add(name)

		input_nodes = {name : network[name] for name in input_names}
		output_nodes = {name : network[name] for name in output_names}

		# Last step: resolve the dependency graph.
		ordered = OrderedDict()
		# Pre-populate with input nodes.
		for k, v in network.items():
			if not v.inputs:
				ordered[k] = v
		for k in ordered:
			del network[k]
		# Now add the rest of the graph.
		while network:
			# Make the next pass through the nodes.
			changed = set()
			for k, v in network.items():
				for node in v.inputs:
					if node.container.name not in ordered:
						break
				else:
					ordered[k] = v
					changed.add(k)
			if not changed:
				raise ValueError('No change during dependency graph '
					'resolution. There is something wrong with the graph.')
			for k in changed:
				del network[k]
		network = ordered

		return input_nodes, output_nodes, network

	###########################################################################
	def enumerate_nodes(self, root):
		""" Enumerates all of the layers in the model.
		"""
		if root.terminal():
			return [CollapsedContainer(
				inputs=root.inputs or [],
				container=root,
				names=[root.name] if root.name else []
			)]

		result = []
		for child in root.get_children(recursive=False):
			result.extend(self.enumerate_nodes(child))
		if result:
			if root.inputs:
				result[0].inputs = root.inputs
			if root.name:
				result[-1].names.append(root.name)
			if root.sink:
				result[-1].sink = True
		return result

	###########################################################################
	def has_extension(self, name):
		""" Returns True if the model has the current extension attached.

			# Arguments

			name: str. The name of the extension to query.
		"""
		return name in self.extensions

	###########################################################################
	def extend(self, name, containers, rebuild=True):
		""" Extends the model with additional nodes.

			# Arguments

			name: str. The name of this extension, which can be used to remove
				it later. If an extension already exists with this name, a
				ValueError exception is raised.
			containers: list of Containers. The containers to extend the model
				with.
			rebuild: bool (default: True). Whether or not to rebuild the model
				after the extension is added. Generally, this is a very good
				idea, and should only be False if `build()` will be called
				explicitly (as when modifying extensions in a batch).
		"""
		if name in self.extensions:
			raise ValueError('An extension with the same name has already '
				'been applied: {}'.format(name))

		self.extensions[name] = containers
		for container in containers:
			self.root.add_child(container)

		if rebuild:
			self.build()

		for extension_callback in self.extension_callbacks:
			extension_callback.extended(name)

	###########################################################################
	def retract(self, name, rebuild=True):
		""" Removes a previously attached model extension.

			# Arguments

			name: str. The name of the extension to remove. If it doesn't
				exist, a ValueError is raised.
			rebuild: bool (default: True). Whether or not to rebuild the model
				after the extension is removed. Generally, this is a very good
				idea, and should only be False if `build()` will be called
				explicitly (as when modifying extensions in a batch).

			# Return value

			The list of Containers that previously extended the model.
		"""
		if name not in self.extensions:
			raise ValueError('No such extension found: {}'.format(name))

		containers = self.extensions.pop(name)
		for container in containers:
			self.root.remove_child(container)

		if rebuild:
			self.build()

		for extension_callback in self.extension_callbacks:
			extension_callback.retracted(name)

		return containers

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

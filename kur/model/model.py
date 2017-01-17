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

		self.compiled = None

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
	def supplement_provider(self, provider):
		for name, source in self.get_data_sources():
			provider.add_source(source, name=name)

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
			logger.debug(
				'No such data source found for shape inference: %s', name)
			return None

		shape = self.provider.sources[index].shape()
		logger.debug('Inferred shape for input "%s": %s', name, shape)
		return shape

	###########################################################################
	def get_data_name_by_layer_name(self, keys, layer_name,
		key_cache=None, aliases=None):
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
		key_cache = key_cache or self.key_cache
		aliases = aliases or (self.input_aliases, self.output_aliases)

		key = key_cache.get(layer_name)
		if key in keys:
			return key

		for alias_list in aliases:
			for key, name in alias_list.items():
				if key in keys and name == layer_name:
					key_cache[layer_name] = key
					return key

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

		self.compiled = None

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
				meat = False

				for layer in node.container.build(self):
					if layer is None:
						continue
					meat = True
					value = self.backend.connect(
						inputs=value,
						target=layer
					)

				if meat:
					node.value = value
				else:
					if len(value) != 1:
						raise ValueError('You cannot have an "empty" layer '
							'with more than one input. The "{}" layer has {} '
							'input layers.'.format(node.container.name,
							len(value)))
					node.value = value[0]
					node.names.extend(node.inputs[0].names)
					node.inputs[0].names = []

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
				outputs[node.container.name] = node
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
		network.by_name = {}
		for node in nodes:
			container_node = ContainerNode(
				inputs=[],
				container=node.container,
				outputs=[],
				names=node.names,
				value=None
			)
			network[node.container.name] = container_node
			for name in node.names:
				network.by_name[name] = container_node

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
		ordered.by_name = network.by_name
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
	def get_shape_at_layer(self, name, assumptions=None):
		return self.get_shape_at_node(
			self.network.by_name[name],
			assumptions or {}
		)

	###########################################################################
	def get_shape_at_node(self, node, assumptions):
		for k, v in assumptions.items():
			if k in node.names:
				return v

		if node.inputs:
			return node.container.shape(
				input_shapes=[
					self.get_shape_at_node(input_node, assumptions)
					for input_node in node.inputs
				]
			)
		else:
			return node.container.shape(None)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

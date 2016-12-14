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
import warnings
from collections import namedtuple, OrderedDict, deque
from ..containers import Container
from ..containers.operators import ContainerGroup
from ..containers.layers import Placeholder
from ..engine import PassthroughEngine

logger = logging.getLogger(__name__)

# Convenience class for keeping track of high-level network nodes.
ContainerNode = namedtuple('ContainerNode',
	['inputs', 'container', 'sink', 'outputs']
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
	def apply_provider_knowledge(self, provider):
		""" Enables inference of tensor shapes by modifying the parsed
			containers given provider information.
		"""
		logger.debug('Applying provider-inferred shapes to input layers.')
		if provider.keys is None:
			logger.debug('No provider keys available. Cannot infer shapes.')
			return

		sources = dict(zip(provider.keys, provider.sources))
		for target in self.root.get_children(
				recursive=True, include_self=True
			):
			if not isinstance(target, Placeholder):
				continue

			logger.debug('Trying to infer shape for input "%s"', target.name)

			if target.name not in sources:
				logger.warning('Could not find a data source for model '
					'input "%s". Maybe you meant one of these: %s',
					target.name, ', '.join(provider.keys))
				continue

			source = sources.pop(target.name)
			shape = source.shape()

			if target.shape is None:
				logger.debug('Inferred shape for input "%s": %s',
					target.name, shape)
				target.set_shape(shape)

			elif target.shape != shape:
				logger.warning('The input placeholder "%s" in the model '
					'has an explicit shape %s which disagrees with the '
					'shape of the corresponding data source %s.',
					target.name, target.shape, source.shape())

			else:
				logger.debug('Input "%s" already has a shape.', target.name)

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
	@staticmethod
	def _assemble(container, inputs, previous, outputs, sink, result):
		""" Enumerates all pieces of the model that must be built.

			This basically enumerates all containers, assigns them their
			resolved names, and tracks their inputs.

			# Arguments

			container: Container instance. The container to assemble
				ContainerNodes for.
			inputs: list or None. If a list, it should be a list of strings
				that indicate which containers are used as input to
				`container`.
			previous: str or None. If not None, then it is the name of the
				most recent container that was assembled, and which would serve
				as input to this container in the absence of any other explicit
				inputs.
			sink: bool. Whether or not the container is a sink.
			outputs: list or None. If a list, it should be a list of strings
				that indicate which names are associated with this
				`container`'s output. There may be multiple names due to nested
				containers.
			result: list. The output list to populate with ContainerNodes.

			# Return value

			The name of the most recently assembled container, suitable for use
			as `previous` in a recurrent/iterative call to `_assemble`.
		"""

		# If the order is `inputs or container.inputs` then the outermost
		# input list is used. If the order is `container.inputs or inputs`
		# then the innermost input list is used.
		inputs = inputs or container.inputs or previous
		if inputs is None:
			inputs = []
		elif not isinstance(inputs, (list, tuple)):
			inputs = [inputs]

		outputs = outputs or []
		if container.name:
			outputs = outputs + [container.name]

		logger.debug('Assembling container: %s', container.name)
		logger.debug('  Inputs: %s', inputs)
		logger.debug('  Outputs: %s', outputs)
		logger.debug('  Previous: %s', previous)

		if container.terminal():
			result.append(ContainerNode(
				inputs=tuple(inputs),			# List of names.
				container=container,			# Container instance.
				sink=sink or container.sink,	# If the container is a sink
				outputs=tuple(outputs)			# List of names.
			))
			return container.name

		else:
			children = list(container.get_children(recursive=False))
			for child in children:
				# Only the very last container should get tagged with the
				# parent container's output name.
				first_child = child == children[0]
				last_child = child == children[-1]
				previous = Model._assemble(
					container=child,
					inputs=inputs if first_child else previous,
					previous=previous,
					outputs=outputs if last_child else [],
					sink=(sink or container.sink) if last_child else False,
					result=result
				)
			return previous

	###########################################################################
	def assemble(self):
		""" Convenience function for constructing all the container-level graph
			nodes.

			# Return value

			A list of ContainerNodes.
		"""
		result = []
		Model._assemble(
			container=self.root,
			inputs=None,
			previous=None,
			outputs=None,
			sink=False,
			result=result
		)
		return result

	###########################################################################
	@staticmethod
	def _attach_within_container(backend, container_nodes):
		""" Instantiates the subgraphs of the low-level network that are
			trivially described within a container.

			# Arguments

			backend: Backend instance. The backend to use for instantiating
				graph elements.
			container_nodes: list of ContainerNodes. The list of assembled
				containers (as returned by `assemble()`) to create connections
				for.

			# Return value

			A tuple `(network, endpoints)`, where `network` is a list of
			partially constructed Nodes describing the network, and `endpoints`
			is a dictionary containing unlinked Nodes. `endpoints` has two
			keys: 'input' and 'output', which indicate whether the nodes
			contained in the respective keys are associated with the input or
			output ends of the container. The values in `endpoint` are also
			dictionaries of Nodes indexed by container name.
		"""

		network = []
		endpoints = {
			'input' : {},
			'output' : {}
		}

		# For each container, assemble its nodes.
		for container_node in container_nodes:
			logger.debug('Processing container node: %s', container_node)

			prev = None
			for layer in container_node.container.build(backend):
				node = Node(
					inputs=[prev] if prev is not None else None,
					operation=layer,
					value=None,
					outputs=[]
				)
				if prev is None:
					endpoints['input'][container_node.container.name] = node
					logger.debug('  Input: %s', node)
				else:
					prev.outputs.append(node)
					logger.debug('  Middle: %s', node)
				prev = node
				network.append(node)
			if prev is not None:
				endpoints['output'][container_node.container.name] = node
				logger.debug('  Output: %s', node)

		return network, endpoints

	###########################################################################
	@staticmethod
	def _attach_between_containers(container_nodes, endpoints):
		""" Finishes initializing low-level graph Nodes by enumerating
			connections between (rather than within) containers.

			# Arguments

			container_nodes: list of ContainerNodes. The ContainerNodes to
				create connections between.
			endpoints: dict. A dictionary describing the partially constructed
				Nodes, as returned by `_attach_within_container`.

			# Return value

			A tuple `(inputs, outputs)` where each of `inputs` and `outputs`
			are `OrderedDict' instances. These ordered dictionaries have keys
			which are the names of the containers associated with the inputs
			and outputs of the network, respectively. The values of these
			dictionaries are the Node instances themselves.
		"""
		# Data structures for holding the return values.
		inputs = OrderedDict()
		outputs = OrderedDict()

		# Find nodes which produce a given output name.
		node_by_output = OrderedDict(
			[(name, node) for node in container_nodes for name in node.outputs]
		)

		# Keep track of which nodes we use as inputs, so that later on we can
		# decide if they are output nodes.
		input_used = set()

		# Loop over each container node.
		for container_node in container_nodes:

			# Grab its name.
			container_name = container_node.container.name

			# Check if we have inputs to attach.
			if container_name not in endpoints['input']:
				warnings.warn('Container name was not found in list of open '
					'inputs. This is a bug, but we can try to skip it.')
				continue

			# Grab the node object.
			node = endpoints['input'][container_name]

			# Figure out where the input comes from.
			if isinstance(container_node.container, Placeholder):
				# Oh, it's just a placeholder. Sweet.
				inputs[container_name] = node
			else:
				# It's another container. Let's connect to each of its inputs.
				prev = []
				for input_name in container_node.inputs:

					# Was this created by another layer?
					if input_name in node_by_output:
						# Yes, attach to it.

						# Get the node
						input_node = node_by_output[input_name]

						# Mark its output as used by someone.
						input_used.add(input_node.container.name)

						# Now grab the node associated with the
						# input container's output.
						input_node = \
							endpoints['output'][input_node.container.name]

						# Add it as an input
						prev.append(input_node)

						# Let the input node know that we are one of its
						#outputs.
						input_node.outputs.append(node)

					else:
						# No other container owns this input. It's probably
						# because it is an implicit input, but without a
						# Placeholder to mark it as such.
						raise ValueError('Placeholder inference is not '
							'supported yet.')

				# Set the node's inputs.
				node.inputs = prev

		# Now figure out which nodes are truly outputs.
		for container_name in endpoints['output']:
			# Grab the node
			node = endpoints['output'][container_name]

			# Grab the container.
			container_node = node_by_output[container_name]

			# To be an output, the container needs to be a sink, or the output
			# has to not be used as an input anywhere.
			if container_node.sink or not container_name in input_used:
				outputs[container_name] = node

		return inputs, outputs

	###########################################################################
	def is_built(self):
		""" Returns True if this model has been built at some point.
		"""
		return self.network is not None

	###########################################################################
	def build(self):
		""" Builds the model.
		"""

		logger.info('Building the model.')

		if not self._parsed:
			logger.warning('The model has not been parsed yet. We will try to '
				'parse it without context, but this may easily fail. Make '
				'sure Model.parse() is called before Model.build().')
			self.parse(None)

		# Construct the high-level network nodes.
		container_nodes = self.assemble()

		# Assemble the in-container nodes.
		network, endpoints = Model._attach_within_container(
			self.backend,
			container_nodes
		)

		if logger.isEnabledFor(logging.DEBUG):

			logger.debug('Network:')
			for entry in network:
				logger.debug('  - %s', entry)

		# Now do the between-container attaching.
		inputs, outputs = Model._attach_between_containers(
			container_nodes,
			endpoints
		)

		logger.info('Model inputs:  %s', ', '.join(node for node in inputs))
		logger.info('Model outputs: %s', ', '.join(node for node in outputs))

		self._connect_network(self.backend, inputs)

		self.network = network
		self.inputs = inputs
		self.outputs = outputs
		self.endpoints = endpoints
		self.container_nodes = container_nodes

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

	###########################################################################
	@staticmethod
	def _connect_network(backend, inputs):
		""" Applies all tensor operations to instantiate a fully-connected
			tensor graph.

			# Arguments

			backend: Backend instance. The backend to use to connect graph
				elements.
			network: list of Nodes. The nodes which constitute the network.
			inputs: list of Nodes. The input nodes of the network.
		"""
		queue = deque(inputs.values())
		while queue:
			node = queue.popleft()

			logger.debug('Connecting node: %s', node)

			if node.inputs is None:
				node.value = node.operation
			else:
				node.value = backend.connect(
					inputs=[x.value for x in node.inputs],
					target=node.operation
				)

			if node.outputs:
				queue.extend(node.outputs)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

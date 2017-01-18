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

import warnings
from . import ParsingError
from ..utils import get_subclasses

###############################################################################
class Container:
	""" The base class for all parseable entries in the model representation.

		# General container features

		- name: unique, immutable (sticks to the first named container)
		- tag: non-unique, mutable (sticks to the last named container)
		- oldest: non-unique, immutable (sticks to the first named container)
		- inputs: the layers that are used as input
		- sink
		- when:
		- skip:
	"""

	# Used for generating unique names.
	counter = {}

	###########################################################################
	@classmethod
	def get_container_name(cls):
		""" Returns the name of the container class.

			This is the name used to determine which container needs to be
			instantiated. It can be overriden in derived classes if the name of
			the class isn't the same as the key used in the Kurfile.

			# Return value

			A lower-case string unique to this container class.
		"""
		return cls.__name__.lower()

	###########################################################################
	@classmethod
	def unique_name(cls):
		""" Generates a unique name for this instance.
		"""
		if cls in Container.counter:
			val = Container.counter[cls]
		else:
			val = 0
		Container.counter[cls] = val + 1
		return '..{}.{}'.format(cls.get_container_name(), val)

	###########################################################################
	@staticmethod
	def create_container_from_data(data, **kwargs):
		""" Factory method for creating containers.
		"""
		cls = Container.find_container_for_data(data)
		return cls(data, **kwargs)

	###########################################################################
	@staticmethod
	def find_container_for_data(data):
		""" Finds a class object corresponding to a data blob.
		"""
		for cls in get_subclasses(Container):
			if cls.get_container_name() in data:
				return cls
		raise ValueError('No such container for data: {}'.format(data))

	###########################################################################
	def __init__(self, data):
		""" Creates a new container.
		"""
		self.data = data
		self.tags = []
		self.name = None
		self.oldest = []
		self.inputs = []
		self.args = None
		self.sink = False

		self.parent = None
		self.children = []

		self._built = None
		self._parsed = False

	###########################################################################
	def __str__(self):
		""" Return a string representation.

			This uses the container's declared name in the representation.
		"""
		return '{type}(name={name})'.format(
			type=self.get_container_name(),
			name=self.name
		)

	###########################################################################
	def __repr__(self):
		""" Return a string representation.

			This uses the container's class name in the representation.
		"""
		return '{type}(name={name})'.format(
			type=self.__class__.__name__,
			name=self.name
		)

	###########################################################################
	def is_parsed(self):
		""" Returns True if this container has been parsed already.
		"""
		return self._parsed

	###########################################################################
	def add_child(self, container):
		""" Attaches an existing container as a child to this container.
		"""
		if container.parent is not None:
			raise ParsingError('Cannot add a child container which already '
				'has parents.')
		container.parent = self
		self.children.append(container)

	###########################################################################
	def remove_child(self, container, recursive=True):
		""" Removes a child container by name.

			# Arguments

			container: str or Container. If a Container instance, the container
				to remove. If a string, the name of the child container to
				remove.
			recursive: bool (default: True). If True, searches recursively for
				the child container; otherwise, only looks at this container's
				immediate children.

			# Return value

			None

			# Exceptions

			If the child container cannot be found, a ValueError exception is
			raised.
		"""
		name = container.name if isinstance(container, Container) \
			else container

		child = self.get_child_by_name(name, recursive=recursive)
		if child is None:
			raise ValueError('No such child found: {}'.format(name))
		target = child.parent
		if target is None:
			raise ValueError('Malformed children. A parent is missing for "{}"'
				.format(child.name))
		child.parent = None
		target.children.remove(child)

	###########################################################################
	def new_child_from_data(self, data):
		""" Creates a new container from data and attaches it as a child.
		"""
		child = Container.create_container_from_data(data)
		self.add_child(child)
		return child

	###########################################################################
	def get_children(self, recursive=False, include_self=False):
		""" Returns this container's children.

			# Arguments

			recursive: bool (default: False). If True, recursively searches
				each container's children for additional children.
			include_self: bool (default: False). If True, yields this container
				as well.

			# Return value

			An iterator over the children of this container.
		"""
		if include_self:
			yield self
		for child in self.children:
			yield child
			if recursive:
				yield from child.get_children(recursive=recursive)

	###########################################################################
	def get_child_by_name(self, name, recursive=True):
		""" Searches for a child container with the given name.

			# Arguments

			name: str. The name of the child container.
			recursive: bool (default: True). If True, each child container's
				children are also searched for the given named child.

			# Return value

			If the named child is found, it is returned. Otherwise, None is
			returned.
		"""
		for child in self.children:
			if child.name == name:
				return child
			elif recursive:
				result = child.get_child_by_name(name, recursive=recursive)
				if result is not None:
					return result
		return None

	###########################################################################
	def get_root(self):
		""" Returns the root node of the parent/child tree.
		"""
		if self.parent:
			return self.parent.get_root()
		else:
			return self

	###########################################################################
	def get_relative_by_name(self, name):
		""" Searches the entire parent/child hierarchy for a container with the
			given name.

			# Arguments

			name: str. The name of the container to search for.

			# Return value

			If the container is found, it is returned; otherwise, None is
			returned.
		"""
		return self.get_root().get_child_by_name(name, recursive=True)

	###########################################################################
	def parse(self, engine):
		""" Convenience function for parsing the container.

			This should not be overriden in derived classes. Override
			`_parse()` instead.
		"""
		if self._parsed:
			return

		self._parse_pre(engine)
		# TODO: Apply any "vars" that got parsed out during `_parse_pre()`.
		self._parse(engine)
		self._parse_post(engine)

		self._parsed = True

	###########################################################################
	def build(self, model, rebuild=False):
		""" Convenience function for building the underlying operations.

			This should not be overriden in derived classes. Override
			`_build()` instead.
		"""
		if not self._parsed:
			raise ParsingError('Container must be parsed before being built.')
		if self._built is None or rebuild:
			self._built = list(self._build(model))
		yield from self._built

	###########################################################################
	def _parse_pre(self, engine):
		""" Pre-parsing hook.

			Specifically, we may need to parse out the "vars" field (if
			present) so that we can do variable substitution during `_parse()`.
		"""
		if isinstance(self.data, str):
			self.data = engine.evaluate(self.data)
			if isinstance(self.data, str):
				self.data = {self.data : None}

	###########################################################################
	def _parse_post(self, engine):
		""" Post-parsing hook.

			The primary purpose of this is to register the container with the
			templating engine so that it can be referenced by other containers
			or Kurfile sections.
		"""
		engine.register(self)

	###########################################################################
	def _parse(self, engine):
		""" Parse the container.

			This should be overriden in derived classes.
		"""
		if 'tag' in self.data:
			self.tags = engine.evaluate(self.data['tag'], recursive=True)
			if not isinstance(self.tags, (list, tuple)):
				self.tags = [self.tags]
		else:
			self.tags = []

		if self.name is None:
			if 'name' in self.data:
				self.name = engine.evaluate(self.data['name'])
			else:
				self.name = self.unique_name()

		if 'oldest' in self.data:
			self.oldest = engine.evaluate(self.data['oldest'], recursive=True)
			if not isinstance(self.oldest, (list, tuple)):
				self.oldest = [self.oldest]
		else:
			self.oldest = []

		container_name = self.get_container_name()
		if container_name in self.data:
			self.args = engine.evaluate(self.data[container_name])
		else:
			if not self.is_anonymous():
				warnings.warn('This container "{}" has no arguments.'
					.format(self.name), SyntaxWarning)
			self.args = None

		if 'inputs' in self.data:
			self.inputs = engine.evaluate(self.data['inputs'], recursive=True)
			if not isinstance(self.inputs, (list, tuple)):
				self.inputs = [self.inputs]
		else:
			self.inputs = []

		if 'sink' in self.data:
			sink = engine.evaluate(self.data['sink'])
			if isinstance(sink, str):
				if sink.lower() in ('yes', 'true', 'on'):
					sink = True
				elif sink.lower() in ('no', 'false', 'off'):
					sink = False
				else:
					raise ParsingError('Cannot evaluate boolean "sink" '
						'string: {}'.format(sink))
			elif not isinstance(sink, bool):
				raise ParsingError('Cannot evaluate boolean "sink": {}'.format(
					sink))

			self.sink = sink
		else:
			self.sink = False

	###########################################################################
	def _build(self, model):
		""" Constructs the backend-specific data objects.

			# Arguments

			model: Model instance. The model to use for constructing the
				container.

			# Return value

			An iterator that yields Layers.

			# Notes

			- If you need to get an instance of the backend while constructing
			  your container, use `model.get_backend()`.
		"""
		raise NotImplementedError

	###########################################################################
	def validate(self, key, required=True, dtype=None, list_like=False,
		raise_error=True):
		""" Checks if a key exists and optionally has the right type.

			# Arguments

			key: object. The key to check for in the dictionary.
			required: bool (default: True). Whether or not the absence of the
				key should be a fatal error.
			dtype: type or tuple of types (default: None). If not None and the
				key exists, then the value is checked to see if its type
				matches `dtype` or, if `dtype` is a tuple, the value is checked
				to see if its type matches one of the types in `dtype`.
			list_like: bool or type or list of types (default: False). Only
				used when `dtype` is not None. If True, then the value is
				assumed to be iterable, and type-checking takes place over each
				element of the iterable rather than to the iterable type
				itself. If `list_like` is a type or list of types, then the
				iterable is also checked to ensure that it is one of the
				supported types, and then `dtype` type-checking still occurs
				over each element of the iterable.
			raise_error: bool (default: True). If True, validation failures
				raise a ParsingError exception; otherwise, a boolean result is
				returned.

			# Return value

			If `raise_error` is False, then True is returned if validation
			passes, and False is returned otherwise.

			# Exceptions

			If the key is required but missing, or if the key is present by has
			an incorrect type, then a ParsingError is thrown if `raise_error`
			is True.
		"""
		name = self.__class__.__name__

		# First check if the key exists.
		if key not in self.data:
			# If the key is not required, then we're done.
			if not required:
				return True

			if raise_error:
				raise ParsingError('Missing key for {}: {}'.format(name, key))
			else:
				return False

		# Now do type checking if requested.
		if dtype is not None:
			if list_like is True:
				if all(isinstance(val, dtype) for val in self.data[key]):
					return True
			elif list_like:
				if isinstance(self.data[key], list_like):
					if all(isinstance(val, dtype) for val in self.data[key]):
						return True
			else:
				if isinstance(self.data[key], dtype):
					return True

			if raise_error:
				raise ParsingError('Bad type for key "{}" in "{}". Expected '
					'{}{} but received {}: {}'.format(key, name,
					'a list of ' if list_like else '',
					dtype, type(self.data[key]), self.data[key])
				)
			else:
				return False

		return True

	###########################################################################
	def terminal(self):
		""" Whether or not this container is responsible for producing backend-
			specific operations.

			# Return value

			If this container actually produces backend-specific operations (as
			opposed to deferring to children for specific layers), or if it
			wraps backend operations with additional operations, then this
			function should return True.

			If this container simply yields its children's containers and
			provides higher-level control of the architecture, then it should
			return False.
		"""
		raise NotImplementedError

	###########################################################################
	def is_anonymous(self):
		""" Whether or not this container is intended to be used by end-users,
			or is merely an internal container used by Kur.
		"""
		return False

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

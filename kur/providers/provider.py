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
from ..utils import get_subclasses

logger = logging.getLogger(__name__)

###############################################################################
class Provider:						# pylint: disable=too-few-public-methods
	""" Base class for all data providers.

		Data providers take a set of data sources and synchronize their inputs,
		providing batches to the network for training. Introducing the Provider
		as an abstraction between a data source and the training loop is
		convenient in order to allow different data sources to produce data at
		different rates (e.g., in-memory, on-disk, over-network).

		From a software design perspective, the Provider is a strategy pattern
		for allowing the training system to retrieve data.

		# Expected behavior

		- Whenever any Source iterator finishes, the entire Provider stops
		  iterating.
		- Whenever a Provider stops iterating, that is the end of the epoch.
		- Whenever a Provider starts iterating (at the beginning of each
		  epoch), it creates new iterators for each Source. Source iterators
		  are not persisted across Provider iterations.
		- Sources are permitted to be different lengths.
		- Because the Provider stops iterating (and terminates the epoch) as
		  soon as the first (shortest) Source completes, and because Sources
		  are allowed to have different lengths, some sources (which are finite
		  but longer than the shortest source) may never have their entire
		  datasets iterated over (because the epoch will end before the last
		  entries in the longer sources are reached). If this is unacceptable,
		  then you should consider restructuring your data sources, or using
		  shuffleable sources with a ShuffleProvider.
		- Sources are said to be infinite if they cannot determine how many
		  samples they can provide (i.e., "len" returns <= 0 for the Source).
		  A Provider is said to be infinite if all of its Sources are infinite.
		  Note that an infinite Provider or Source does not need to never
		  terminate; it simply means that it cannot determine when the iterator
		  will finish. An infinite Source can still stop iterating, causing the
		  infinite Provider to complete data providing for the epoch.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the provider.

			# Return value

			A lower-case string unique to this provider.
		"""
		return cls.__name__.lower()

	###########################################################################
	@staticmethod
	def get_all_providers():
		""" Returns an iterator to the names of all providers.
		"""
		for cls in get_subclasses(Provider):
			yield cls

	###########################################################################
	@staticmethod
	def get_provider_by_name(name):
		""" Finds a provider class with the given name.
		"""
		name = name.lower()
		for cls in Provider.get_all_providers():
			if cls.get_name() == name:
				return cls
		raise ValueError('No such provider with name "{}"'.format(name))

	###########################################################################
	def __init__(self, sources):
		""" Create a new data provider.

			# Arguments

			sources: dict or list. If this is a list, then it is a list of
				Sources. If this is a dictionary, its values are Sources and
				its keys are string names that are used by the training process
				to determine which nodes in the network should receive which
				information.
		"""
		if isinstance(sources, list):
			self.keys = None
			self.sources = sources
		elif isinstance(sources, dict):
			self.keys = list(sources.keys())
			self.sources = [sources[k] for k in self.keys]
		else:
			raise ValueError('Unknown or unexpected type for "sources": {}'
				.format(type(sources)))

		self._calculate_sizes()

		if self.keys is None:
			for source in self.sources:
				if source.is_derived() and source.requires():
					# In principle, we could support this by giving a list of
					# all the data to these sources, but that sounds hazardous.
					raise ValueError('Anonymous providers cannot have named, '
						'dependent sources.')

		for source in self.sources:
			source.on_added(self)

	###########################################################################
	def __contains__(self, key):
		if self.keys is None:
			raise ValueError('Cannot use "in" with a key-less provider.')
		return key in self.keys

	###########################################################################
	def source_shapes(self):
		""" Prints debug information about the sources in this provider.
		"""
		if logger.isEnabledFor(logging.DEBUG):
			for i, source in enumerate(self.sources):
				if self.keys is None:
					name = "anonymous"
				else:
					name = self.keys[i]

				logger.debug('Data source "%s": entries=%s, shape=%s',
					name, len(source), source.shape())

	###########################################################################
	def _calculate_sizes(self):
		""" Calculates the sizes of each of the sources.
		"""
		lens = [len(source) for source in self.sources]
		len_set = {i for i in lens if i > 0}
		if len(len_set) == 1:
			# All finite sources are the same length.
			self.entries = len_set.pop()
		elif len_set:
			# Finite sources have differing lengths.
			self.entries = min(lens)
			if logger.isEnabledFor(logging.INFO):
				if self.keys is None:
					msg = 'Sizes are: {}'.format(
						', '.join(str(x) for x in lens)
					)
				else:
					msg = 'Sizes are: {}'.format(
						', '.join('{}={}'.format(k, v)
							for k, v in zip(self.keys, lens)
						)
					)
				logger.info('Mixed source lengths. Make sure you know what '
					'you are doing. %s', msg)
		else:
			# There are no finite sources.
			self.entries = 0
		self.lens = lens

	###########################################################################
	def has_source(self, name):
		if name is None or self.keys is None:
			raise ValueError('Cannot check for anonymous sources.')
		try:
			return self.keys.index(name)
		except ValueError:
			return False

	###########################################################################
	def add_source(self, source, name=None):
		""" Adds a new data source to an existing provider.

			# Arguments

			source: Source instance. The data source to add.
			name: str or None (default: None). The name of the data source.
		"""
		if name is None or self.keys is None:
			if source.is_derived() and source.requires():
				raise ValueError('Anonymous providers cannot have named, '
					'dependent sources.')
		else:
			if self.has_source(name):
				return

		if name is None:
			self.keys = None
		elif self.keys is not None:
			self.keys.append(name)
		self.sources.append(source)
		self._calculate_sizes()

		source.on_added(self)

	###########################################################################
	def pre_iter(self):
		""" This is a hook that is called whenever the provider begins
			generating data for the next epoch.
		"""
		pass

	###########################################################################
	def __iter__(self):
		""" Return an iterator which can generate batches of data.

			The iterator should return a tensor of shape `(X, ) +
			self.shape()`, where `X` is number of entries in the batch returned
			by this provider (it is implementation-specific, but will be used
			by the training/testing/evauation process as a batch).

			# Notes:

			- The implementation must call `pre_iter()` as the first call in
			  `__iter__`.
		"""
		raise NotImplementedError

	###########################################################################
	def wrap(self, data):
		""" Wraps data up in the same way it was presented to the Provider.

			# Arguments

			data: list. A list of tensors, one for each data source.

			# Return value

			If `sources` in `__init__` was a list, returns a list of tensors
			in the same order. If `sources` was a dict, returns a dictionary
			with the same keys.
		"""
		if self.keys:
			return {k : v for k, v in zip(self.keys, data)}
		else:
			return data

	###########################################################################
	def __len__(self):
		""" Returns the total number of entries that this provider returns per
			epoch.

			# Return value

			If the provider is finite (because at least one of its sources is
			finite), then the provider must return the number of samples it
			provides per epoch (that is, the length of the shortest finite
			source).

			If the provider is infinite (because all of its sources are
			infinite), then it should return <= 0.
		"""
		return self.entries

	###########################################################################
	def source_dependencies(self):
		dependencies = {}
		for i, source in enumerate(self.sources):
			x = []
			if source.is_derived():
				if source.requires() and self.keys is None:
					raise ValueError('Anonymous providers cannot have derived '
						'sources which have named requirements.')
				for k in source.requires():
					if k not in self.keys:
						raise ValueError('Tried to resolve source '
							'dependencies, but we cannot find this source: {}'
							.format(k))
					x.append(self.keys.index(k))
			dependencies[i] = x
		return dependencies

	###########################################################################
	def order_sources(self):
		result = [
			i for i, source in enumerate(self.sources)
			if not source.is_derived()
		] + [
			i for i, source in enumerate(self.sources)
			if source.is_derived() and not source.requires()
		]

		remaining = set(
			i for i, source in enumerate(self.sources)
			if source.is_derived() and source.requires()
		)

		if not remaining:
			return result

		# At this point, only derived sources with named dependencies have yet
		# to be resolved.

		if self.keys is None:
			raise ValueError('Anonymous providers cannot have derived '
				'sources which have named requirements. This should have '
				'already been checked, so this is definitely a bug.')

		# Get the dependency tree.
		dependencies = {}
		for i in remaining:
			depends = set()
			for k in self.sources[i].requires():
				if k not in self.keys:
					raise ValueError('Tried to resolve source '
						'dependencies, but we cannot find this source: {}'
						.format(k))
				depends.add(self.keys.index(k))
			dependencies[i] = depends

		while dependencies:
			changed = False
			for this_source, needs_these_sources in list(dependencies.items()):
				if any(source in dependencies \
					for source in needs_these_sources):
					# Can't use this yet.
					continue

				result.append(this_source)
				del dependencies[this_source]
				changed = True
			if not changed:
				raise ValueError('We are stuck in a circular dependency '
					'while trying to process derived, named sources. Make '
					'sure to eliminate circular dependencies; if you cannot '
					'find any, this may be a bug.')

		return result

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

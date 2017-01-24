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

import numpy

from . import Provider

logger = logging.getLogger(__name__)

###############################################################################
class ShuffleProvider(Provider): \
	# pylint: disable=too-few-public-methods,abstract-method
	""" Data provider that can shuffle data in-sync with support data sources.

		# Expected behavior

		This extends the list of expected behaviors from the `Provider` base
		class.

		- Because Providers create new Source iterators for each epoch, and
		  because each epoch ends whenever any Source iterator terminates (that
		  is, once the shorted dataset terminates), Providers only shuffle
		  Sources once every epoch, and can do so at the beginning of each
		  epoch. There is no need for Sources to indicate when they would like
		  to be shuffled.
		- There are two possibilities when applying shuffles:

			1. Past shuffles that have been applied to Sources are persistent,
			   affecting the Source's data in a non-reverseable way (as far as
			   the Source is concerned).
			2. Shuffles are temporary, and once the Source iterator is
			   recreated (between epochs), it can be assumed that the data---
			   prior to another shuffle---is in the same, original ordering.

		  These distinctions are transparent to the provider, which merely
		  needs to provide new shuffle indices. But it is possible for the
		  provider to help orchestrate these (e.g.. by providing reverse
		  permutations after each epoch).

		  Now, all Sources must be Shuffleable in order to use shuffling. This
		  means that even mixed finite/infinite Sources must support the
		  Shuffleable protocol. Sources that stream data samples (e.g.,
		  real-time data augmentation generators) could possibly be simplified
		  if the shuffle is assumed to be "new" after each epoch. Indeed, there
		  may be performance benefits (cache-coherency) from making certain
		  assumptions about how the shuffle "ought" to be applied.

		  Despite this, it would be imprudent to decide that Shuffleable
		  sources must operate under any particular set of assumptions about
		  how they store or organize data. Therefore, we opt for (1) and assume
		  that shuffles are persistent, and leave it to the individual sources
		  to decide how to combine multiple shuffles.
	"""

	###########################################################################
	def __init__(self, *args, randomize=True, sort_by=None, sortagrad=None,
		shuffle_after=None, **kwargs):
		""" Create a new data provider that can shuffle shuffleable sources.

			# Arguments

			sources: dict or list. If this is a list, then it is a list of
				Sources. If this is a dictionary, its values are Sources and
				its keys are string names that are used by the training process
				to determine which nodes in the network should receive which
				information.
		"""
		super().__init__(*args, **kwargs)

		if randomize:
			for i, source in enumerate(self.sources):
				if not source.can_shuffle():
					raise ValueError('All data sources must be shuffleable '
						'for the provider to able to shuffle them. Source '
						'"{}" does not seem to match: {}'.format(
							'unknown' if self.keys is None else self.keys[i],
							source))

			if randomize is True:
				if self.entries <= 0:
					raise ValueError('Must know how long our shuffleable '
						'sources are in order to shuffle them, but all '
						'sources seem to be infinite. If this is the case, '
						'then set `randomize` to an integer in the '
						'Provider\'s constructor, or disable shuffling '
						'entirely by setting `randomize` to False.')
				self._shuffle_len = self.entries
			elif isinstance(randomize, int):
				self._shuffle_len = randomize # pylint: disable=redefined-variable-type
			else:
				raise ValueError('`randomize` must be True/False or an '
					'integer, but we received: {}'.format(randomize))

			self.randomize = True
		else:
			self.randomize = False

		if sortagrad:
			if sort_by or shuffle_after:
				raise ValueError('"sortagrad" cannot be used with "sort_by" '
					'or "shuffle_after". That is because sortagrad=X is '
					'equivalent to sort_by=X, shuffle_after=1.')
			sort_by = sortagrad
			shuffle_after = 1

		if sort_by:
			if self.keys is None:
				raise ValueError('Cannot use "sort_by" with unnamed sources.')
			try:
				sort_data = self.sources[self.keys.index(sort_by)]
			except ValueError:
				raise ValueError('Could not find the "sort_by" key "{}" in '
					'list of available sources: {}'
					.format(sort_by, ', '.join(self.keys)))

			if len(sort_data) <= 0:
				raise ValueError('Data sorting requires a finite source.')
		else:
			sort_data = None
		self.sort_by = sort_by
		self.sort_data = sort_data
		self.is_sorted = False

		self.shuffle_after = shuffle_after or 0

	###########################################################################
	def add_source(self, source, name=None):
		""" Adds a new data source to an existing provider.
		"""
		if self.randomize:
			if not source.can_shuffle():
				raise ValueError('Cannot add a non-shuffleable source to an '
					'already shuffled provider.')

		super().add_source(source, name=name)

		if self.randomize is True:
			self._shuffle_len = self.entries

	###########################################################################
	def pre_iter(self):
		""" Pre-iteration hook.

			This is our opportunity to shuffle the data sources.
		"""
		super().pre_iter()

		if self.randomize:
			if self.shuffle_after > 0:
				self.shuffle_after -= 1

				if self.sort_by and not self.is_sorted:
					logger.info('Sorting data by key %s...', self.sort_by)
					n = numpy.empty(
						(len(self.sort_data), ) + self.sort_data.shape()
					)
					start = 0
					for batch in self.sort_data:
						n[start:start+len(batch)] = batch[:]
						start += len(batch)
					indices = numpy.argsort(n)
					for source in self.sources:
						source.shuffle(indices)

					self.is_sorted = True
				else:
					logger.info('Suppressing shuffle...')
			else:
				logger.debug('Shuffling...')
				indices = numpy.random.permutation(self._shuffle_len)
				for source in self.sources:
					source.shuffle(indices)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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

###############################################################################
class Source:						# pylint: disable=too-few-public-methods
	""" Base class for all data sources.

		Data sources are responsible for producing data that will be presented
		to the model for training, validation, or prediction (evaluation).

		Each data source provides one of the input or output tensors that the
		network needs.
	"""

	###########################################################################
	def __init__(self):
		""" Create a new data source.
		"""
		pass

	###########################################################################
	def __iter__(self):
		""" Return an iterator to the data.

			The iterator should return a tensor of shape `(X, ) +
			self.shape()`, where `X` is the number of entries provided by this
			data source.

			# Note

			If `is_derived()` is False, then this should simply `yield` data.
			If `is_derived()` is True, then Kur will set up a bi-directional
			generator with two `yield`s expected. The first `yield` will give
			input to the Source, and the second `yield` is for returning the
			data back to Kur.
		"""
		raise NotImplementedError

	###########################################################################
	def __len__(self):
		""" Returns the total number of entries that this source can return, if
			known.

			# Return value

			If the total number of entries that this source can return is
			known, it should be returned. In the simplest case, this is the
			same as the number of training samples in an epoch. If this number
			is unknown or effectively infinite (because it is from some
			real-time data generator or is being streamed from some
			uncontrolled source, for example), then this should return <= 0.
		"""
		raise NotImplementedError

	###########################################################################
	def shape(self):
		""" Return the shape of the tensor (excluding batch size) returned by
			this data source.

			# Return value

			A shape tuple (length of each dimension in the tensor).
		"""
		raise NotImplementedError

	###########################################################################
	def on_added(self, provider):
		""" Called when this source is added to a provider.
		"""
		pass

	###########################################################################
	def is_derived(self):
		""" Returns True if this source is a derived source (requires other
			sources to be present in a provider in order to function).
		"""
		raise NotImplementedError

	###########################################################################
	def can_shuffle(self):
		""" Returns True if this data source can be shuffled.

			By default, sources are not shuffleable.
		"""
		raise False

	###########################################################################
	def shuffle(self, indices):
		""" Applies a permutation to the data.

			# Arguments

			indices: numpy array. List of indices to use in constructing the
				data permutation.

			# Return value

			None
		"""
		raise NotImplementedError

###############################################################################
class OriginalSource(Source):				# pylint: disable=abstract-method
	""" A source which is not derived.
	"""

	###########################################################################
	def is_derived(self):
		""" By definition, this source is not derived.
		"""
		return False

###############################################################################
class DerivedSource(Source):	# pylint: disable=abstract-method
	""" A source which is not derived.
	"""

	###########################################################################
	def is_derived(self):
		""" By definition, this source is derived.
		"""
		return True

	############################################################################
	def requires(self):
		""" Returns a tuple of data sources that this source requires.
		"""
		raise NotImplementedError

	###########################################################################
	def __iter__(self):
		""" Derived sources take two steps: a `yield` to get input, and `yield`
			to send output. We provide a default implementation that is likely
			suitable for many derived sources.
		"""
		self.setup()
		while True:
			inputs = yield
			outputs = self.derive(inputs)
			yield outputs

	###########################################################################
	def __len__(self):
		""" Returns the total number of entries that this source can return.
		"""
		return 0

	###########################################################################
	def setup(self):
		""" Called at the beginning of iteration.
		"""
		pass

	###########################################################################
	def derive(self, inputs):
		""" Compute the derived source given its inputs.

			# Arguments

			inputs: list of arrays. Each list corresponds to the respective
				data source specified in `requires()`.

			# Return value

			An array that is passed back to the data provider.
		"""
		raise NotImplementedError

	###########################################################################
	def shuffle(self, indices):
		""" Applies a permutation to the data.
		"""
		pass

	###########################################################################
	def can_shuffle(self):
		""" Returns True if this data source can be shuffled.
		"""
		return True

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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

from ..utils import get_subclasses

###############################################################################
class Loss:
	""" Base class for all loss functions (also called objective functions).
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the loss function.

			# Return value

			A lower-case string unique to this loss function.
		"""
		return cls.__name__.lower()

	###########################################################################
	@staticmethod
	def get_all_losses():
		""" Returns an iterator to the names of all loss functions.
		"""
		for cls in get_subclasses(Loss):
			yield cls

	###########################################################################
	@staticmethod
	def get_loss_by_name(name):
		""" Finds a loss function class with the given name.
		"""
		name = name.lower()
		for cls in Loss.get_all_losses():
			if cls.get_name() == name:
				return cls
		raise ValueError('No such loss function with name "{}"'.format(name))

	###########################################################################
	def __init__(self, weight=None):
		""" Creates a new loss function.

			# Arguments

			weight: float. A relative weight to apply to this loss function.
				This is only meaningful in models which have multiple loss
				functions.
		"""
		self.weight = 1.0 if weight is None else weight

	###########################################################################
	def get_weight(self):
		""" Returns the loss function weight.
		"""
		return self.weight

	###########################################################################
	def get_loss(self, backend):
		""" Returns the loss function that can be used by the implementation-
			specific model.
		"""
		raise NotImplementedError

	###########################################################################
	def modify(self, model, name): \
		# pylint: disable=unused-argument,no-self-use
		""" Modify/extend the model to fit this loss function.

			Some loss functions will want to modify the model in some way in
			order to properly instrument the model. For example, CTC loss is a
			little different in the sense that it wants additional inputs at
			training and evaluation time. Many loss functions, however, will
			not need this functionality.

			# Arguments

			model: Model instance. The model to modify.
			name: str. The output of the model to modify.

			# Return value

			The name of the model output to apply this loss function to.
			Normally, this is the same as `name`, but if the loss function
			modifies the model, then it may need to act on a different layer
			instead.
		"""
		return name

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

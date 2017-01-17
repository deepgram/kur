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
def keras_wrap(model, target, output, loss):
	""" Convenience function for wrapping a Keras loss function.
	"""
	# pylint: disable=import-error
	import keras.objectives as O
	import keras.backend as K
	# pylint: enable=import-error
	if isinstance(loss, str):
		loss = O.get(loss)
	shape = model.outputs[target].value._keras_shape # pylint: disable=protected-access
	ins = [
		(target, K.placeholder(
			ndim=len(shape),
			dtype=K.dtype(model.outputs[target].value),
			name=target
		))
	]
	out = loss(ins[0][1], output)
	return ins, out

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
		if weight is not None:
			raise ValueError('Loss function weights have not been implemented '
				'yet.')
		self.weight = 1.0 if weight is None else weight

	###########################################################################
	def get_weight(self):
		""" Returns the loss function weight.
		"""
		return self.weight

	###########################################################################
	def get_loss(self, model, target, output):
		""" Returns the loss tensor for this output.

			# Arguments

			model: Model instance.
			target: str. The name of the output layer to apply the loss
				function to.
			output: tensor (implemented-specific). The symbolic tensor for this
				output layer.

			# Return value

			A tuple of the form:

			```python
			(
				# Input tensors
				[
					(input_name, placeholder),
					(input_name, placeholder),
					...
				],

				# Output value
				loss_value
			)
			```

			The derived class is required to return all required input
			placeholder, including placeholders for the target model outputs.
		"""
		raise NotImplementedError

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

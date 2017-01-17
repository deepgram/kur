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
def keras_clip(optimizer):
	""" Convenience function for creating kwargs for setting Keras gradient
		clipping.
	"""
	if optimizer.clip_type is None:
		return {}
	elif optimizer.clip_type == 'norm':
		return {'clipnorm' : optimizer.clip_value}
	elif optimizer.clip_type == 'abs':
		return {'clipvalue' : optimizer.clip_value}
	else:
		raise ValueError('Unhandled clip type: {}. This is a bug.'
			.format(optimizer.clip_type))

###############################################################################
def keras_wrap(keras_optimizer):
	""" Convenience function for wrapping up a Keras optimizer function for use
		by Kur.
	"""
	def optimize(weights, loss):
		""" Returns the update tensor.

			# Arguments

			weights: weight tensor
			loss: loss tensor
		"""
		return keras_optimizer.get_updates(weights, [], loss)
	return optimize

###############################################################################
class Optimizer:
	""" Base class for all optimizers
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the optimizer.

			# Return value

			A lower-case string unique to this optimizer.
		"""
		return cls.__name__.lower()

	###########################################################################
	@staticmethod
	def get_all_optimizers():
		""" Returns an iterator to the names of all optimizers.
		"""
		for cls in get_subclasses(Optimizer):
			yield cls

	###########################################################################
	@staticmethod
	def get_optimizer_by_name(name):
		""" Finds a optimizer class with the given name.
		"""
		name = name.lower()
		for cls in Optimizer.get_all_optimizers():
			if cls.get_name() == name:
				return cls
		raise ValueError('No such optimizer with name "{}"'.format(name))

	###########################################################################
	def __init__(self, clip=None):
		""" Creates a new optimizer.
		"""
		if clip is None:
			self.clip_type = None
		elif isinstance(clip, dict):
			if 'norm' in clip:
				self.clip_type = 'norm'
				self.clip_value = clip['norm']
			elif 'abs' in clip:
				self.clip_type = 'abs'
				self.clip_value = clip['abs']
			else:
				raise ValueError('Unknown clip spec: {}'.format(clip))

			try:
				self.clip_value = float(self.clip_value)
			except ValueError:
				raise ValueError('Invalid clip value: {}'
					.format(self.clip_value))
		else:
			raise ValueError('"clip" expected a dictionary with key "norm" or '
				'"value". Instead, we got: {}'.format(clip))

	###########################################################################
	def get_optimizer(self, backend):
		""" Returns the optimizer that can be used by the implementation-
			specific model.
		"""
		raise NotImplementedError

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

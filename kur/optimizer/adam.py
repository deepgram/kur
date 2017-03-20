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

from . import Optimizer, keras_clip, keras_wrap

###############################################################################
class Adam(Optimizer):
	""" Adam optimizer
	"""

	###########################################################################
	def __init__(self, learning_rate=None, decay=None, *args, **kwargs):
		""" Create a new Adam optimizer.

			# Arguments

			learning_rate: float. The learning rate to use.
		"""
		super().__init__(*args, **kwargs)

		self.learning_rate = learning_rate or 0.001
		self.decay = decay or 0
		self.optimizer = None

	###########################################################################
	def get_optimizer(self, backend):
		""" Returns a backend-specific instantiation of the optimizer.
		"""
		if backend.get_name() == 'keras':
			import keras.optimizers as O		# pylint: disable=import-error
			self.optimizer = self.optimizer or O.Adam(
				lr=self.learning_rate,
				decay=self.decay,
				**keras_clip(self)
			)
			return keras_wrap(self.optimizer)
		elif backend.get_name() == 'pytorch':
			import torch.optim as optim			# pylint: disable=import-error
			if self.optimizer is None:
				self.optimizer = lambda params: optim.Adam(
					params,
					lr=self.learning_rate,
					weight_decay=self.decay
				)
			return self.optimizer
		else:
			raise ValueError('Unsupported backend "{}" for optimizer "{}"'
				.format(backend.get_name(), self.get_name()))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

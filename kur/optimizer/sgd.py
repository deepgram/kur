"""
Copyright 2017 Deepgram

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

from . import Optimizer, keras_clip, keras_wrap

logger = logging.getLogger(__name__)

###############################################################################
class SGD(Optimizer):
	""" Stochastic gradient descent optimizer
	"""

	###########################################################################
	def __init__(self, learning_rate=None, momentum=None, decay=None,
		nesterov=None, *args, **kwargs):
		""" Create a new Adam optimizer.

			# Arguments

			learning_rate: float. The learning rate to use.
			momentum: float. Momentum for parameter updates
			decay: float learning rate decay over each update
			nesterov: bool. Whether or not to apply Nesterov momentum.
		"""
		super().__init__(*args, **kwargs)

		self.learning_rate = learning_rate or 0.01
		self.momentum = momentum or 0.0
		self.decay = decay or 0.0
		self.nesterov = nesterov or False

		self.optimizer = None

	###########################################################################
	def get_optimizer(self, backend):
		""" Returns a backend-specific instantiation of the optimizer.
		"""
		if backend.get_name() == 'keras':
			import keras.optimizers as O		# pylint: disable=import-error
			self.optimizer = self.optimizer or O.SGD(
				lr=self.learning_rate,
				momentum=self.momentum,
				decay=self.decay,
				nesterov=self.nesterov,
				**keras_clip(self)
			)
			return keras_wrap(self.optimizer)
		elif backend.get_name() == 'pytorch':
			import torch.optim as optim			# pylint: disable=import-error
			if self.nesterov:
				logger.warning('The PyTorch backend does not use Nesterov '
					'momentum. Ignoring it...')
			if self.optimizer is None:
				self.optimizer = lambda params: optim.SGD(
					params,
					lr=self.learning_rate,
					momentum=self.momentum,
					weight_decay=self.decay
				)
			return self.optimizer
		else:
			raise ValueError('Unsupported backend "{}" for optimizer "{}"'
				.format(backend.get_name(), self.get_name()))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

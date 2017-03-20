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

from . import Loss, keras_wrap

###############################################################################
class MeanSquaredError(Loss):
	""" Mean squared error: a simple loss function to minimize distance between
		the model output and the true value.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the loss function.
		"""
		return 'mean_squared_error'

	###########################################################################
	def get_loss(self, model, target, output):
		backend = model.get_backend()

		if backend.get_name() == 'keras':
			return keras_wrap(model, target, output, 'mean_squared_error')
		elif backend.get_name() == 'pytorch':

			# pylint: disable=import-error
			import torch.nn as nn
			# pylint: enable=import-error

			loss = model.data.move(nn.MSELoss())

			return [
				[
					(target, model.data.placeholder(target))
				],
				lambda inputs, output: loss(
					output, inputs[0]
				)
			]
		else:
			raise ValueError('Unsupported backend "{}" for loss function "{}"'
				.format(backend.get_name(), self.get_name()))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

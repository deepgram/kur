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
class CategoricalCrossentropy(Loss):
	""" Categorical crossentropy loss, used for 1-of-N classification.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the loss function.
		"""
		return 'categorical_crossentropy'

	###########################################################################
	def get_loss(self, model, target, output):
		backend = model.get_backend()

		if backend.get_name() == 'keras':
			return keras_wrap(model, target, output, 'categorical_crossentropy')
		elif backend.get_name() == 'pytorch':

			# pylint: disable=import-error
			import torch
			import torch.nn as nn
			# pylint: enable=import-error

			loss = model.data.move(nn.NLLLoss())

			def do_loss(truth, prediction):
				""" Calculates CCE loss.
				"""

				# Truth will be one-hot: (batch_size, ..., n_words)
				# But PyTorch only uses class labels (rather than one-hot).
				# PyTorch doesn't automatically broadcast loss into higher
				# dimensions, so we need to flatten it out.

				# There is only one input for this loss function.
				truth = truth[0]

				# Flatten it out into: (lots of entries, number of classes)
				truth = truth.view(-1, truth.size(truth.dim() - 1))
				# Convert one-hot to class label: (lots of entries, )
				truth = torch.max(truth, 1)[1].squeeze(1)

				# Flatten out the prediction into:
				#   (lots of entries, number of classes)
				prediction = prediction.view(
					-1,
					prediction.size(prediction.dim() - 1)
				)

				return loss(prediction, truth)

			return [
				[
					(target, model.data.placeholder(target))
				],
				do_loss
			]
		else:
			raise ValueError('Unsupported backend "{}" for loss function "{}"'
				.format(backend.get_name(), self.get_name()))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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

from . import Layer

###############################################################################
class BatchNormalization(Layer):	# pylint: disable=too-few-public-methods
	""" A batch normalization layer, which normalizes activations.
	"""

	###########################################################################
	@classmethod
	def get_container_name(cls):
		""" Returns the name of the container class.
		"""
		return 'batch_normalization'

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new batch normalization layer.
		"""
		super().__init__(*args, **kwargs)
		self.dimension = None

	###########################################################################
	def _build(self, model):
		""" Instantiates the layer with the given backend.
		"""
		backend = model.get_backend()
		if backend.get_name() == 'keras':

			import keras.layers as L			# pylint: disable=import-error
			yield L.BatchNormalization(
				mode=2,
				axis=-1
			)

		else:
			raise ValueError(
				'Unknown or unsupported backend: {}'.format(backend))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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

from .. import Container

###############################################################################
class Layer(Container):						# pylint: disable=abstract-method
	""" Base class for layers, which are containers that produce
		backend-specific layers, or which wrap backend-specific layers with
		additional backend-specific operations.
	"""

	###########################################################################
	def terminal(self):
		""" The identifying feature of a Layer is that it is terminal, meaning
			that it actually produces layers.
		"""
		return True

	###########################################################################
	def shape(self, input_shapes):
		""" Returns the output shape of this layer for a given input shape.

			# Arguments

			input_shape: list of tuples. A list of tuples, corresponding to the
				shapes of the input layers, one for each input layer, excluding
				batch size.

			# Return value

			A tuple specifying the shape of each output from the layer,
			excluding batch size.
		"""
		raise NotImplementedError

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

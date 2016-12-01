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

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

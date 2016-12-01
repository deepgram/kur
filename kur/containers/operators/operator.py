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
class Operator(Container):					# pylint: disable=abstract-method
	""" Base class for operators, which are containers that do not produce
		backend-specific layers, but rather defer to their children to do so.
	"""

	###########################################################################
	def terminal(self):
		""" The identifying feature of an Operator is that it is non-terminal,
			so we implement the base-class method here.
		"""
		return False

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

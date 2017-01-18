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

from . import Operator

###############################################################################
class ContainerGroup(Operator):		# pylint: disable=too-few-public-methods
	""" A pseudo-container for holding other containers.

		The primary purpose for this container is to provide a unified
		namespace for all containers. This way, even if the model is specified
		over over multiple Containers, this ContainerGroup will still allow
		containers to "find" other containers.
	"""

	###########################################################################
	def __init__(self, containers):
		""" Creates a new container group.

			# Arguments

			containers: list of Containers. The containers to add as children.

			# Notes

			None of the child containers should already have a parent; if any
			of them do, a ParsingError is raised.
		"""
		super().__init__(data=None)

		for child in containers:
			self.add_child(child)

	###########################################################################
	def _parse(self, engine):
		""" Parse all children.
		"""
		for child in self.children:
			child.parse(engine)

	###########################################################################
	def _build(self, model):
		""" Build all children.
		"""
		for child in self.children:
			yield from child.build(model)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

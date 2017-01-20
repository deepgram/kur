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

import os

###############################################################################
class EnvironmentalVariable:		# pylint: disable=too-few-public-methods
	""" Context management for modifying environmental variables.
	"""

	###########################################################################
	def __init__(self, **kwargs):
		""" Creates a new EnvironmentalVariable manager.

			# Arguments

			**kwargs: key/value pairs to use in setting environmental variables
				inside the context.
		"""
		self.vars = kwargs
		self.old_vars = None

	###########################################################################
	def __enter__(self):
		""" Enters the context, setting the temporary environmental variables.
		"""
		self.push()

	###########################################################################
	def push(self):
		""" Sets the environmental variables. Idempotent.
		"""
		if self.old_vars is not None:
			return

		self.old_vars = {}
		for k, v in self.vars.items():
			self.old_vars[k] = os.environ.get(k)
			if v is None:
				if k in os.environ:
					del os.environ[k]
			else:
				os.environ[k] = v

	###########################################################################
	def __exit__(self, exc_type, exc_value, traceback):
		""" Exists the context, restoring the original environmental variables.
		"""
		self.pop()

	###########################################################################
	def pop(self):
		""" Reets the environmental variables. Idempotent.
		"""
		if self.old_vars is None:
			return

		for k, v in self.old_vars.items():
			if v is None:
				if k in os.environ:
					del os.environ[k]
			else:
				os.environ[k] = v

		self.old_vars = None

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

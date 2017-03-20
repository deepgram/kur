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

import logging
import contextlib
import sys

###############################################################################
def redirect_stderr(x):
	""" Redirects stderr to another file-like object.

		This is some compatibility code to support Python 3.4.
	"""
	if hasattr(contextlib, 'redirect_stderr'):
		result = contextlib.redirect_stderr
	else:
		@contextlib.contextmanager
		def result(x):
			""" Stand-in for Python 3.5's `redirect_stderr`.

				Notes: Non-reentrant, non-threadsafe
			"""
			old_stderr = sys.stderr
			sys.stderr = x
			yield
			sys.stder = old_stderr

	return result(x)

###############################################################################
class DisableLogging:				# pylint: disable=too-few-public-methods
	""" Context manager that disables logging temporarily.
	"""

	###########################################################################
	def __init__(self, level=logging.DEBUG):
		""" Suppresses all logging at `level` and below.
		"""
		self.level = level

	###########################################################################
	def __enter__(self):
		""" Start suppression.
		"""
		logging.disable(self.level)

	###########################################################################
	def __exit__(self, exc_type, exc_val, exc_tb):
		""" Resume regular logging.
		"""
		logging.disable(logging.NOTSET)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

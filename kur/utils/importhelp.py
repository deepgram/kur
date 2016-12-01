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

import importlib

def can_import(name, package=None):
	""" Returns True if the given module can be imported.

		# Arguments

		name: str. The name of the module.
		package: str. The name of the package, if `name` is a relative import.
			This is ignored for Python versions < 3.4.

		# Return value

		If importing the specified module should succeed, returns True;
		otherwise, returns False.
	"""
	try:
		importlib.util.find_spec
	except AttributeError:
		# Python < 3.4
		return importlib.find_loader(		# pylint: disable=deprecated-method
			name
		) is not None
	else:
		# Python >= 3.4
		return importlib.util.find_spec(name, package=package) is not None

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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
import ast
import json
import logging

import yaml
import jinja2

from .engine import Engine
from ..utils import CudaContext, CudaError

logger = logging.getLogger(__name__)

###############################################################################
def combine(value, new=None):
	""" Jinja2 filter which merges dictionaries.
	"""
	new = new or {}
	value = dict(value)
	value.update(new)
	return value

###############################################################################
def as_dict(value, key):
	""" Jinja2 filter which constructs a dictionary from a key/value pair.
	"""
	return {key : value}

###############################################################################
def ternary(value, result_true, result_false):
	""" Implements a ternary if/else conditional.
	"""
	return result_true if value else result_false

###############################################################################
# pylint: disable=protected-access
def gpu_count():
	""" Returns the number of GPU devices available on the system.

		Notes
		-----

		The result of this function is cached so that it will return
		immediately during future calls.
	"""
	if gpu_count._value is None:
		try:
			with CudaContext() as context:
				gpu_count._value = len(context)
		except CudaError:
			gpu_count._value = 0
	return gpu_count._value
gpu_count._value = None
# pylint: enable=protected-access

###############################################################################
def resolve_path(engine, filename):
	""" Resolves a path relative to the Kurfile.
	"""
	filename = os.path.expanduser(os.path.expandvars(filename))

	kurfile = engine._scope.get('filename')
	if kurfile:
		filename = os.path.join(
			os.path.dirname(kurfile),
			filename
		)

	return os.path.abspath(filename)

###############################################################################
def create_load_json(engine):
	""" Creates the JSON loader.
	"""

	# pylint: disable=protected-access
	def load_json(filename, use_cache=True):
		""" Loads a JSON file from disk.
		"""
		path = resolve_path(engine, filename)
		logger.debug('Loading JSON file: %s (%s)', filename, path)
		if use_cache and path in load_json.cache:
			logger.trace('Using cached data.')
		else:
			with open(path) as fh:
				load_json.cache[path] = json.loads(fh.read())
		return load_json.cache[path]
	load_json.cache = {}
	# pylint: enable=protected-access

	return load_json

###############################################################################
def create_load_yaml(engine):
	""" Creates the YAML loader.
	"""

	# pylint: disable=protected-access
	def load_yaml(filename, use_cache=True):
		""" Loads a YAML file from disk.
		"""
		path = resolve_path(engine, filename)
		logger.debug('Loading YAML file: %s (%s)', filename, path)
		if use_cache and path in load_yaml.cache:
			logger.trace('Using cached data.')
		else:
			with open(path) as fh:
				load_yaml.cache[path] = yaml.load(fh.read())
		return load_yaml.cache[path]
	load_yaml.cache = {}
	# pylint: enable=protected-access

	return load_yaml

###############################################################################
class JinjaEngine(Engine):
	""" An evaluation engine which uses Jinja2 for templating.
	"""

	###########################################################################
	def register_custom_filters(self, env):
		""" Adds our custom filters to the Jinja2 engine.

			Arguments
			---------

			env: jinja2.Environment instance. The environment to add the custom
				filters to.
		"""
		env.filters['basename'] = os.path.basename
		env.filters['dirname'] = os.path.dirname
		env.filters['splitext'] = os.path.splitext
		env.filters['combine'] = combine
		env.filters['as_dict'] = as_dict
		env.filters['ternary'] = ternary

		env.globals['gpu_count'] = gpu_count
		env.globals['load_json'] = create_load_json(self)
		env.globals['load_yaml'] = create_load_yaml(self)

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new Jinja2 templating engine.
		"""

		# Call the parent
		super().__init__(*args, **kwargs)

		# Create a Jinja2 environment. We could use jinja2.Template().render()
		# directly, but having an environment gives us more control over, e.g.,
		# custom filters.
		self.env = jinja2.Environment()

		# Registering custom filters is described here:
		#   http://jinja.pocoo.org/docs/dev/api/#custom-filters
		self.register_custom_filters(self.env)

		# Built-in Jinja2 filters are listed here:
		#   http://jinja.pocoo.org/docs/dev/templates/#builtin-filters

	###########################################################################
	def _evaluate(self, expression):
		""" Evaluates an expression in the current scope.

			# Arguments

			expression: str. The string to evaluate.

			# Return value

			The evaluated expression (some Python object/class).
		"""
		result = self.env.from_string(expression).render(**self._scope)

		# Jinja2's `render()` will return a string which is a valid Python
		# expression (e.g., passing it through `eval` will succeed). However,
		# if you reference, e.g., a list that Jinja renders, the list will get
		# printed as a string. So we use `ast.literal_eval()` to turn it back
		# into a Python object. This may have unintended consequences, such as
		# turning the literal string "None" into the `None` Python object.
		# But it's better than nothing.
		try:
			result = ast.literal_eval(result)
		except (ValueError, SyntaxError):
			pass

		return result

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

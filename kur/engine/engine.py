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

from collections import ChainMap
import warnings
import logging

logger = logging.getLogger(__name__)

###############################################################################
class ScopeStack:					# pylint: disable=too-few-public-methods
	""" Context management for Engine scopes.

		# Example

		```python
		engine = Engine(...)
		with ScopeStack(engine, {'key' : 'value'}):
			pass
		```

		Or use multiple scopes:
		```python
		engine = Engine(...)
		scope1 = {'key' : 'value'}
		scope2 = {'key2' : 'value2'}
		with ScopeStack(engine, (scope1, scope2)):
			pass
		```
	"""

	###########################################################################
	def __init__(self, engine, scope):
		""" Creates a new scope stack.

			# Arguments

			engine: Engine instance. The engine to manage scope for.
			scope: dictionary or list of dictionaries. The scope(s) to add to
				the engine's scope stack during context management.
		"""
		self.engine = engine
		if not isinstance(scope, (list, tuple)):
			scope = [scope]
		self.scope = scope

	###########################################################################
	def __enter__(self):
		""" Enter context management and add scopes to the engine.
		"""
		for scope in self.scope:
			self.engine.scope(**scope)
		return self.engine

	###########################################################################
	def __exit__(self, exc_type, exc_value, traceback):
		""" Leave context management and pop the scopes.
		"""
		for _ in self.scope:
			self.engine.scope_pop()

###############################################################################
class Engine:
	""" Base class for all template engines.
	"""

	###########################################################################
	def __init__(self):
		""" Creates a new engine

			# Arguments

			scope: dict or None. The default key/value scope to use, or None to
				begin with an empty scope.
		"""
		self.state = {
			'tags' : {},
			'oldest': {},
			'layers' : []
		}
		self._scope = ChainMap(self.state)

	###########################################################################
	def scope(self, **kwargs):
		""" Create an additional scope.

			# Arguments

			kwargs: dict. The key/value scope to augment the current scope
				with.

			# Return value

			Returns the current instance (self).

			# Notes

			- This is intended to be used with context management, like so:

				```python
				engine = Engine()
				with engine.scope(a='b', c='d'):
					engine.evaluate(...)
				```

			- It can also be used standalone to add context layers, like so:

				```python
				engine = Engine()
				engine.scope(a='b', c='d')
				```
		"""
		self._scope = self._scope.new_child(kwargs)
		return self

	###########################################################################
	def scope_pop(self):
		""" Removes the most recent scope.
		"""
		self._scope = self._scope.parents

	###########################################################################
	def __enter__(self):
		""" Enter context management.
		"""
		return self

	###########################################################################
	def __exit__(self, exc_type, exc_value, traceback):
		""" Exit context management, popping the most recent scope.
		"""
		self.scope_pop()

	###########################################################################
	def register(self, container):
		""" Adds information about a Container to the templating engine.

			# Arguments

			container: Container instance. The container to register.
		"""
		# Tags
		for tag in container.tags:
			self.state['tags'][tag] = container.name

		# First-names
		for oldest in container.oldest:
			if oldest not in self.state['oldest']:
				self.state['oldest'] = oldest

		# Layers themselves
		self.state['layers'].append(container.name)

	###########################################################################
	def evaluate(self, expression, recursive=False):
		""" Evaluates an expression in the current scope.

			# Arguments

			expression: object. The object to evaluate. If it is a string, then
				it should be evaluated for template substitution and returned.
				Otherwise, the behavior depends on `recursive`. If `recursive`
				is True, then container types(dict, list, tuple), are
				recursively evaluated, preserving the original Python structure
				as much as possible. If `recursive` is False, or if the
				expression in not a container type, the expression is returned
				unchanged.
			recursive: bool (default: False). If True, container types are
				evaluated recursively; otherwise, container types are returned
				unchanged.

			# Return value

			The evaluated expression (some Python object/class).
		"""
		if isinstance(expression, (str, bytes)):
			return self._evaluate(expression)
		elif isinstance(expression, (int, float, complex, type(None))):
			pass
		elif recursive:
			if isinstance(expression, dict):
				return {k : self.evaluate(v, recursive=recursive)
					for k, v in expression.items()}
			elif isinstance(expression, list):
				return [self.evaluate(x, recursive=recursive)
					for x in expression]
			elif isinstance(expression, tuple):
				return tuple(self.evaluate(x, recursive=recursive)
					for x in expression)
			else:
				warnings.warn('Unexpanded Python type {}: {}'.format(
					type(expression),
					expression
				), UserWarning)
		return expression

	###########################################################################
	def _evaluate(self, expression):
		""" Evaluates a string expression in the current scope.

			# Arguments

			expression: str. The string to evaluate.

			# Return value

			The evaluated expression (some Python object/class)
		"""
		raise NotImplementedError

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

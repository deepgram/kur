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

from collections import OrderedDict

###############################################################################
def deep_merge(*args, strategy=None):
	""" Merges data structures together.

		# Arguments

		args: tuple. The list of data structures that should be merged.
		strategy: str or None (default: None). The strategy to use in merging
			the data structures. If None, it defaults to "blend". See
			`Strategies` below.

		# Strategies

		- merge: dictionaries are merged recursively. This means that if a key
		  is not present in both dictionaries, then the key and its value are
		  included in the merged result. If the key is in both dictionaries,
		  its value is recursively merged if both values are themselves
		  dictionaries, but only the last value is kept if the types are
		  different.
		- blend: dictionaries are merged recursively (as with "merge").
		  List/tuples are also merged in a similar manner, elementwise.
		- concat: dictionaries are merged recursively (as with "merge").
		  List/tuples are concatenated.
	"""

	func = {
		'merge' : _merge,
		'blend' : _blend,
		'concat' : _concat
	}.get(strategy or 'blend')
	if func is None:
		raise ValueError('Invalid strategy: {}'.format(strategy))

	if not args:
		return args
	elif len(args) == 1:
		return args[0]
	else:
		result = args[0]
		for x in args[1:]:
			result = func(result, x)
		return result

###############################################################################
def _blend(x, y):								# pylint: disable=invalid-name
	""" Implements the "blend" strategy for `deep_merge`.
	"""
	if isinstance(x, (dict, OrderedDict)):
		if not isinstance(y, (dict, OrderedDict)):
			return y
		return _merge(x, y, recursion_func=_blend)

	if isinstance(x, (list, tuple)):
		if not isinstance(y, (list, tuple)):
			return y
		result = [_blend(*i) for i in zip(x, y)]
		if len(x) > len(y):
			result += x[len(y):]
		elif len(x) < len(y):
			result += y[len(x):]
		return result

	return y

###############################################################################
def _concat(x, y):								# pylint: disable=invalid-name
	""" Implements the "concat" strategy for `deep_merge`.
	"""
	if isinstance(x, (dict, OrderedDict)):
		if not isinstance(y, (dict, OrderedDict)):
			return y
		return _merge(x, y, recursion_func=_concat)

	if isinstance(x, (list, tuple)):
		if not isinstance(y, (list, tuple)):
			return y
		return [i for L in (x, y) for i in L]

	return y

###############################################################################
def _merge(x, y, recursion_func=None):			# pylint: disable=invalid-name
	""" Implements the "merge" strategy for `deep_merge`.
	"""
	recursion_func = recursion_func or _merge

	if not any(isinstance(i, (dict, OrderedDict)) for i in (x, y)):
		return y

	result = {}

	for k, v in x.items():
		if k in y:
			result[k] = recursion_func(v, y[k])
		else:
			result[k] = v

	for k, v in y.items():
		if k not in x:
			result[k] = v

	return result

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

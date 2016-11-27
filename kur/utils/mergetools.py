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

################################################################################
def deep_merge(*args, strategy=None):
	""" Merges data structures together.

		# Arguments

		args: tuple. The list of data structures that should be merged.
		strategy: str or None (default: None). The strategy to use in merging
			the data structures. If None, it defaults to "merge". See
			`Strategies` below.

		# Strategies

		- merge: dictionaries are merged recursively. This means that if a key
		  is not present in both dictionaries, then the key and its value are
		  included in the merged result. If the key is in both dictionaries, its
		  value is recursively merged if both values are themselves
		  dictionaries, but only the last value is kept if the types are
		  different.
		- blend: dictionaries are merged recursively (as with "merge").
		  List/tuples are also merged in a similar manner.
		- concat: dictionaries are merged recursively (as with "merge").
		  List/tuples are concatenated.
	"""

	func = {
		'merge' : _merge,
		'blend' : _blend,
		'concat' : _concat
	}.get(strategy or 'merge')
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

################################################################################
def _blend(x, y):
	""" Implements the "blend" strategy for `deep_merge`.
	"""
	raise NotImplementedError

################################################################################
def _concat(x, y):
	""" Implements the "concat" strategy for `deep_merge`.
	"""
	raise NotImplementedError

################################################################################
def _merge(x, y):
	""" Implements the "merge" strategy for `deep_merge`.
	"""
	if not any(isinstance(i, (dict, OrderedDict)) for i in (x, y)):
		return y

	result = {}

	for k, v in x.items():
		if k in y:
			result[k] = _merge(v, y[k])
		else:
			result[k] = v

	for k, v in y.items():
		if k not in x:
			result[k] = v

	return result

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

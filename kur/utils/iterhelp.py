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
from concurrent.futures import ThreadPoolExecutor

###############################################################################
def get_any_value(x):
	""" Returns a value of a dict, list, or tuple.

		# Arguments

		x: object. The object to grab a value for.

		# Return value

		If `x` is a dictionary (including an OrderedDict), returns a value from
		it. If `x` is a list or tuple, returns an element from it. Otherwise,
		raises ValueError.
	"""
	if isinstance(x, (dict, OrderedDict)):
		for v in x.values():
			return v
	if isinstance(x, (list, tuple)):
		return x[0]
	raise ValueError('Unexpected data type: {}'.format(type(x)))

###############################################################################
def get_any_key(x):
	""" Returns a key from a dictionary.

		# Arguments:

		x: dict. The dictionary to pull a key from.

		# Return value

		If x is empty, returns None; otherwise, this returns a key from the
		dictionary.
	"""
	if x:
		for k in x:
			return k
	else:
		return None

###############################################################################
def merge_dict(*args):
	""" Merges any number of dictionaries into a single dictionary.

		# Notes

		In Python 3.5+, you can just do this:
		```python
		r = {**x, **y}
		```
		But if you want a single expression in Python 3.4 and below:
		```python
		r = merge_dict(x, y)
		```
	"""
	result = {}
	for x in args:
		result.update(x)
	return result

###############################################################################
def partial_sum(x):
	""" Returns an iterator over the partial sums in a given iterable.

		# Arguments

		x: iterable. The iterable to return partial sums for.

		# Return value

		At each iteration, yields the most recent partial sum, beginning with
		`x[0]`.
	"""
	result = 0
	for i in x:
		result += i
		yield result

###############################################################################
class parallelize:

	def __init__(self, it):
		self.it = it

	def _next(self, it):
		return next(it)

	def __iter__(self):
		pool = ThreadPoolExecutor(1)
		it = iter(self.it)

		future = pool.submit(self._next, it)
		while True:
			result = future.result()
			future = pool.submit(self._next, it)
			yield result

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

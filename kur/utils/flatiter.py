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

###############################################################################
def flatten(x):
	""" Returns an iterator over flattened entries of a list-like data
		structure.
	"""
	for item in x:
		try:
			iter(item)
			if isinstance(item, (str, bytes)):
				yield item
			else:
				yield from flatten(item)
		except TypeError:
			yield item

###############################################################################
def concatenate(args):
	""" Returns an iterator which steps through top-level elements in the
		iterable.
	"""
	for sublist in args:
		yield from iter(sublist)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

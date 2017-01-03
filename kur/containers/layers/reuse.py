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

from . import Layer
from . import ParsingError

###############################################################################
class Reuse(Layer):						# pylint: disable=too-few-public-methods
	""" Reuse weights/operations from another container, rather than creating
		new tensor operations. This allows for weight-sharing between
		containers.

		# Example

		```
		reuse:
		  target: NAME
		```
	"""

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Create a new reuse container.
		"""
		super().__init__(*args, **kwargs)
		self.target = None
		self.target_container = None
		self.reentrant = False

	###########################################################################
	def _parse(self, engine):
		""" Parse the debug statement and print it.
		"""
		super()._parse(engine)

		if isinstance(self.args, dict):
			if 'target' not in self.args:
				raise ParsingError('Missing key "target" in Reuse layer.')
			self.target = engine.evaluate(self.args['target'])
		elif isinstance(self.args, str):
			self.target = engine.evaluate(self.args)
		else:
			raise ParsingError('Reuse layer expected either a dictionary or '
				'a string for an argument, but we received: {}'
				.format(self.args))

	###########################################################################
	def _build(self, model):
		""" Return a reference to another layer.
		"""
		if self.reentrant:
			raise ParsingError('Circular reference found while building reuse '
				'layer: {}'.format(self.name))

		target = self.get_relative_by_name(self.target)
		if target is None:
			raise ParsingError(
				'No such reference found for reuse: {}'.format(self.target))

		if target._built is None:			# pylint: disable=protected-access
			self.reentrant = True
			target.build(model)
			self.reentrant = False

		self.target_container = target
		yield from target.build(model)

	###########################################################################
	def shape(self, input_shapes):
		""" Returns the output shape of this layer for a given input shape.
		"""
		if self.target_container is None:
			raise ValueError('Cannot compute shape for unresolved target.')
		cur_shape = input_shapes
		for child in self.target_container.get_children(
			recursive=True, include_self=True
		):
			if child.terminal():
				cur_shape = [child.shape(cur_shape)]
		return cur_shape[0]

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

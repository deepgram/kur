"""
Copyright 2017 Deepgram

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

from . import Operator
from . import ParsingError

###############################################################################
class ForEach(Operator):				# pylint: disable=too-few-public-methods
	""" Range-based for loops.
	"""

	###########################################################################
	@classmethod
	def get_container_name(cls):
		""" Returns the name of the container class.
		"""
		return 'for_each'

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Create a new for loop.
		"""
		super().__init__(*args, **kwargs)
		self.items = None
		self.loop_var = None

	###########################################################################
	def _parse(self, engine):
		""" Parse and construct the child containers.
		"""
		# Parse self
		if 'items' not in self.args:
			raise ParsingError('Missing "items" key in for_each loop.')
		self.items = engine.evaluate(self.args['items'])

		if 'loop_var' not in self.args:
			self.loop_var = 'item'
		else:
			self.loop_var = engine.evaluate(self.args['loop_var'])

		if 'iterate' not in self.args:
			raise ParsingError('Missing "iterate" key in for_each loop.')
		target = engine.evaluate(self.args['iterate'])

		# Parse children
		for item in range(self.items):
			item = engine.evaluate(item, recursive=True)
			with engine.scope(**{self.loop_var : item}):
				for entry in target:
					self.new_child_from_data(entry).parse(engine)

	###########################################################################
	def _build(self, model):
		""" Construct each child.
		"""
		for child in self.children:
			yield from child.build(model)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

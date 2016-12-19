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

from . import Operator
from . import ParsingError

###############################################################################
class ForLoop(Operator):				# pylint: disable=too-few-public-methods
	""" A classic "for" loop for creating more complex or arbitrary length
		models.

		# Example

		```
		for:
		  range: 10
		  with_index: i
		  iterate:
		    - debug: "Hello {{ i }}"
		```
	"""

	###########################################################################
	@classmethod
	def get_container_name(cls):
		""" Returns the name of the container class.

			Obviously, "for" is an overloaded word in programming. So rather
			than risk problems with statements like `import for`, it is better
			to just have a different name that is used in parsing containers.
		"""
		return 'for'

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Create a new for loop.
		"""
		super().__init__(*args, **kwargs)
		self.limit = None
		self.index = None

	###########################################################################
	def _parse(self, engine):
		""" Parse and construct the child containers.
		"""

		# Always call the parent.
		super()._parse(engine)

		# Parse self
		if 'range' not in self.args:
			raise ParsingError('Missing "range" key in for loop.')
		self.limit = engine.evaluate(self.args['range'])
		try:
			self.limit = int(self.limit)
		except ValueError:
			raise ParsingError('"limit" in a "for" loop must evaluate to an '
				'integer; got this instead: {}'.format(self.limit))

		if 'with_index' not in self.args:
			self.index = 'index'
		else:
			self.index = engine.evaluate(self.args['with_index'])

		if 'iterate' not in self.args:
			raise ParsingError('Missing "iterate" key in for loop.')
		target = engine.evaluate(self.args['iterate'])

		# Parse children
		for index in range(self.limit):
			with engine.scope(**{self.index : index}):
				for entry in target:
					self.new_child_from_data(entry).parse(engine)

	###########################################################################
	def _build(self, model):
		""" Construct each child.
		"""
		for child in self.children:
			yield from child.build(model)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

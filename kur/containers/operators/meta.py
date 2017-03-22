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

###############################################################################
class Meta(Operator):
	""" A meta container is a placeholder for user-defined containers, also
		called templates, which act as macros.
	"""

	###########################################################################
	def is_anonymous(self):
		""" This container's "name" is dynamic, so we need to act as if we have
			no arguments.
		"""
		return True

	###########################################################################
	def _resolve(self, engine):
		""" Figure out which template we represent.
		"""
		name = None
		template = None
		for key in self.data:
			candidate = engine.get_template(key)
			if candidate is None:
				continue
			if template is not None:
				raise ValueError('Ambiguous template substitution. Could be '
					'either "{}" or "{}".'.format(key, name))
			template = candidate
			name = key

		if template is None:
			raise ValueError('Failed to find an appropriate template for '
				'container: {}'.format(self.data))

		return name, template

	###########################################################################
	def _parse(self, engine):
		""" Parses out the meta-container.
		"""

		# Always call the parent.
		super()._parse(engine)

		# Figure out which template we represent.
		name, template = self._resolve(engine)
		self.args = engine.evaluate(self.data[name])

		if self.inputs:
			inputs = self.inputs
		else:
			inputs = engine.state.layers[-1]

		# Put the arguments in scope for the children.
		with engine.scope(args=self.args, inputs=inputs):
			with engine.scope(**self.args):

				# Parse every child.
				for entry in template:
					entry = engine.evaluate(entry)
					self.new_child_from_data(entry).parse(engine)

	###########################################################################
	def _build(self, model):
		""" Construct each child.
		"""
		for child in self.children:
			yield from child.build(model)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

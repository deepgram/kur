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

from . import Layer, ParsingError

################################################################################
class Parallel(Layer):					# pylint: disable=too-few-public-methods
	""" A container for applying tensor operations to each element in a list.
		This is effectively a map operation, like Theano's `scan` or Keras's
		`TimeDistributed`.

		# Example

		```
		parallel:
		  apply:
		    - dense:
		        size: 512
			- convolution:
			    kernels: 16
				...
		```
	"""

	############################################################################
	def __init__(self, *args, **kwargs):
		""" Create a new parallel container.
		"""
		super().__init__(*args, **kwargs)

	############################################################################
	def _parse(self, engine):
		""" Parse the child containers
		"""

		# Always call the parent.
		super()._parse(engine)

		# Parse self
		if 'apply' not in self.args:
			raise ParsingError('Missing "apply" key in parallel container.')
		target = engine.evaluate(self.args['apply'])

		# Parse children
		for child in target:
			self.new_child_from_data(child).parse(engine)

	############################################################################
	def _build(self, backend):
		""" Instantiate the container.
		"""
		for child in self._children:
			for layer in child.build(backend):

				if backend.get_name() == 'keras':

					import keras.layers as L	# pylint: disable=import-error
					yield L.TimeDistributed(
						layer,
						name='{}_{}'.format(self.name, child.name)
					)

				else:
					raise ValueError(
						'Unknown or unsupported backend: {}'.format(backend)
					)

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

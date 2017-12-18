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

import logging

import numpy

from . import DerivedSource

logger = logging.getLogger(__name__)

###############################################################################
class UnzipSource(DerivedSource):

	###########################################################################
	def __init__(self, source_name, target, force_numpy=False):

		super().__init__()

		self.source_name = source_name
		self.target = target
		self.force_numpy = force_numpy

	############################################################################
	def requires(self):
		""" Returns a tuple of data sources that this source requires.
		"""
		return (self.source_name, )

	###########################################################################
	def derive(self, inputs):
		""" Compute the derived source given its inputs.
		"""
		try:
			r = [x[self.target] for x in inputs[0]]
			if self.force_numpy:
				r = numpy.array(r)
			return r
		except IndexError:
			logger.error('Failed to unzip input data: %s', inputs)
			raise

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

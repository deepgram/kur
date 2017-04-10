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

import enum

###############################################################################
class Statistic:

	class Type(enum.Enum):
		BATCH = 1
		TRAINING = 2
		VALIDATION = 3

		@classmethod
		def from_string(cls, value):
			return cls.__members__[value.upper()]

		def __str__(self):
			return self.name.lower()

	def __init__(self, data_type, tags, name):
		if isinstance(data_type, str):
			data_type = Statistic.Type.from_string(data_type)
		self.data_type = data_type

		if tags is None:
			tags = ()
		elif isinstance(tags, str):
			tags = (tags, )
		else:
			tags = tuple(tags)
		self.tags = tags

		self.name = name

	def copy(self, *, data_type=None, tags=None, name=None):
		""" Clones the statistic, optionally changing some of its properties.
		"""
		return Statistic(
			data_type or self.data_type,
			tags or self.tags,
			name or self.name
		)

	def __repr__(self):
		return 'Statistic(data_type="{}", tags={}, name="{}")'.format(
			self.data_type, self.tags, self.name
		)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

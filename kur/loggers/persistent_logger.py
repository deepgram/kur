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

from .logger import Logger

###############################################################################
class PersistentLogger(Logger):
	""" A class for storing log data in a fast binary format that can be
		quickly appended to or randomly seeked.
	"""

	###########################################################################
	def get_clocks(self):
		""" Returns historical clock information.

			Return value
			------------

			If historical clock information exists, it is returned as a
			dictionary. Otherwise, None is returned.
		"""
		return self.clocks

	###########################################################################
	def get_best_training_loss(self):
		""" Returns the best historical training loss.

			# Return value

			If historical training loss is present in the log, it is returned.
			Otherwise, None is returned.
		"""
		raise NotImplementedError

	###########################################################################
	def get_best_validation_loss(self):
		""" Returns the best historical validation loss.

			# Return value

			If historical validation loss is present in the log, it is returned.
			Otherwise, None is returned.
		"""
		raise NotImplementedError

	###########################################################################
	def enumerate_statistics(self):
		""" Returns the available statistics.
		"""
		raise NotImplementedError

	###########################################################################
	def load_statistic(self, statistic):
		""" Loads a particular statistic.

			# Return value

			A tuple of `(batch, timestamp, value)`. Each entry in the tuple may
			be None. All non-None entries will be numpy arrays of the same
			length, containing the respective data.
		"""
		raise NotImplementedError

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

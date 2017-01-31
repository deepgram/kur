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

import pickle
import logging
from . import EvaluationHook

logger = logging.getLogger(__name__)

###############################################################################
class OutputHook(EvaluationHook):
	""" Evaluation hook for saving to disk.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the evaluation hook.
		"""
		return 'output'

	###########################################################################
	@staticmethod
	def _save_as_pickle(target, data, truth=None):
		""" Saves a file as a Python 3 pickle.
		"""
		logger.info('Saving model output as pickle: %s', target)
		result = {'result' : data}
		if truth is not None:
			result['truth'] = truth
		with open(target, 'wb') as fh:
			pickle.dump(result, fh)

	###########################################################################
	def __init__(self, path=None, format=None, **kwargs): \
		# pylint: disable=redefined-builtin
		""" Creates a new output hook.
		"""

		super().__init__(**kwargs)

		if path is None:
			raise ValueError('No path specified in output hook.')

		self.path = path

		format = format or 'pkl'
		savers = {
			'pkl' : OutputHook._save_as_pickle,
			'pickle' : OutputHook._save_as_pickle
		}

		self.saver = savers.get(format)
		if self.saver is None:
			raise ValueError('No such handler for file format: {}'.format(
				format))

	###########################################################################
	def apply(self, current, original, model=None):
		""" Applies the hook to the data.
		"""
		data, truth = current
		self.saver(
			target=self.path,
			data=data,
			truth=truth
		)
		return current

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

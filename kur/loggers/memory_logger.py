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

import os
import re
import shutil
import logging

import yaml

from .logger import Logger
from .statistic import Statistic
from ..utils import idx

logger = logging.getLogger(__name__)

###############################################################################
class MemoryLogger(Logger):
	""" A class for storing log data in-memory. It is non-persistent, but
		useful for tracking session statistics.
	"""

	##########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the logger.
		"""
		return 'memory'

	###########################################################################
	def process(self, data, data_type, tag=None):
		""" Processes training statistics.
		"""
		pass

	###########################################################################
	def __init__(self):
		super().__init__(keep_batch=False, rate=None)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

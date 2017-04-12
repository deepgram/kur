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

import logging
TRACE_LEVEL = 5
def _trace(self, message, *args, **kwargs):
	""" Writes a trace-level message to the log.
	"""
	if self.isEnabledFor(TRACE_LEVEL):
		self._log(TRACE_LEVEL, message, args, **kwargs)
logging.addLevelName(TRACE_LEVEL, 'TRACE')
logging.TRACE = TRACE_LEVEL
logging.Logger.trace = _trace

__homepage__ = 'https://kur.deepgram.com'
from .version import __version__

from . import utils
from . import backend
from . import containers
from . import engine
from . import reader
from . import loss
from . import optimizer
from . import sources
from . import providers
from . import model
from . import supplier
from . import loggers
from .kurfile import Kurfile

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

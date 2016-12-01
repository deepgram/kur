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

import signal
import logging

logger = logging.getLogger(__name__)

###############################################################################
class CriticalSection:				# pylint: disable=too-few-public-methods
	""" Protects critical code from system signals (e.g., keyboard interrupts).

		# Usage

		Prevent KeyboardInterrupt until after the critical section:
		```python
		with CriticalSection():
			# Some code ...

			# This interrupt will be suppressed.
			raise KeyboardInterrupt

			# This code will continue to execute
			# ...

		# Once out of the critical section, the suppressed interrupt will be
		# re-raised.
		```
	"""

	###########################################################################
	def __init__(self, signals=None):
		if signals is None:
			signals = signal.SIGINT

		if not isinstance(signals, (list, tuple)):
			signals = [signals]		# pylint: disable=redefined-variable-type

		self.signals = signals
		self.old_handlers = {}
		self.received = {}

	###########################################################################
	def _handle(self, sig, *args):
		if sig in self.received:
			self.received[sig].append(args)
			logger.debug('Signal received in critical section: %s', sig)
		else:
			logger.warning('Unexpected signal received: %s', sig)

	###########################################################################
	def __enter__(self):
		""" Enters a critical section.
		"""

		self.old_handlers = {}
		for sig in self.signals:

			def handler(sig=sig, *args):
				""" Signal handler (to avoid cell variable problems)
				"""
				self._handle(sig, *args)

			self.received[sig] = []
			self.old_handlers[sig] = signal.signal(
				sig,
				handler
			)

	###########################################################################
	def __exit__(self, exc_type, exc_value, traceback):
		""" Leaves a critical section.
		"""

		# Restore the handlers
		for sig, handler in self.old_handlers.items():
			signal.signal(sig, handler)

		# Call the handlers if necessary.
		for sig, handler in self.old_handlers.items():
			if handler:
				for args in self.received[sig]:
					handler(*args)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

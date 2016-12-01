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

import time

###############################################################################
class Timer:
	""" A simple timing class for gathering performance metrics.
	"""

	###########################################################################
	def __init__(self, started=True):
		""" Creates a new Timer.
		"""
		self.duration = 0
		self.mark = 0
		self.started = False
		if started:
			self.restart()

	###########################################################################
	def resume(self):
		""" Starts a previously paused timer.

			This has no effect if the timer was not paused.
		"""
		if self.started:
			return
		self.started = True
		self.mark = self._clock()

	###########################################################################
	def reset(self):
		""" Reset the current timer (zeros the accumulated time) without
			changing its running state.
		"""
		self.duration = 0

	###########################################################################
	def restart(self):
		""" Restarts a timer (zeros the accumulated time and starts the clock
			from now.
		"""
		self.reset()
		self.started = True
		self.mark = self._clock()

	###########################################################################
	def pause(self):
		""" Pauses a running clock.

			This has no effect if the timer was already paused.
		"""
		if not self.started:
			return
		self.duration += self._clock() - self.mark
		self.started = False

	###########################################################################
	def _clock(self):							# pylint: disable=no-self-use
		""" Returns an internal counter used for timing.
		"""
		return time.perf_counter()

	###########################################################################
	def __call__(self):
		""" Convenience method for getting the current accumulated time.
		"""
		return self.get()

	###########################################################################
	def get(self):
		""" Gets the current accumulated time (has no effect on running state).
		"""
		if self.started:
			return self.duration + (self._clock() - self.mark)
		else:
			return self.duration

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

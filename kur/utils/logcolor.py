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

# Get the eight standard colors.
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# ANSI escape sequences for colored outputs.
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;{}m"
BOLD_SEQ = "\033[1m"

# Color codes for each log-level.
COLORS = {
	'DEBUG': BLUE,
	'INFO': WHITE,
	'WARNING': YELLOW,
	'ERROR': RED,
	'CRITICAL': RED
}

###############################################################################
def basicConfig(**kwargs):						# pylint: disable=invalid-name
	""" A convenience function which mimics the behavior of
		`logging.basicConfig()`, but creates a color formatter.
	"""
	logger = logging.getLogger()
	if logger.hasHandlers():
		return

	if sum(k in kwargs for k in ('filename', 'handlers', 'stream')) > 1:
		raise ValueError('Only one of "filename", "stream", and "handlers" '
			'can be specified.')

	# Create the formatter
	formatterArgs = {							# pylint: disable=invalid-name
		k : kwargs[k] for k in ('datefmt', 'style') if k in kwargs
	}
	if 'format' in kwargs:
		formatterArgs['fmt'] = kwargs['format']
	formatter = ColorFormatter(**formatterArgs)

	# Create a new handler that writes to stderr, and attach the
	# color formatter to it.
	if 'filename' in kwargs:
		handlers = [logging.FileHandler(
			kwargs['filename'],
			mode=kwargs.pop('filemode', 'a')
		)]
	elif 'handlers' in kwargs:
		handlers = kwargs['handlers']
	else:
		handlers = [logging.StreamHandler(stream=kwargs.pop('stream', None))]

	for handler in handlers:
		handler.setFormatter(formatter)
		logger.addHandler(handler)

	# Set the log-level
	if 'level' in kwargs:
		logger.setLevel(kwargs['level'])

###############################################################################
class ColorFormatter(logging.Formatter):
	""" A formatter for Python's logging framework which can produce colored
		logging statements.

		# Usage

		Normally, a basic logger can be configured with:

		```python
		import logging

		logging.basicConfig(
			level=(logging.DEBUG if args.verbose else logging.INFO),
			format='[{asctime} {name}:{lineno}] {message}',
			style='{'
		)
		```

		For colored output, the custom formatter must be created. This
		requires a more complex setup:

		```python
		import logging
		from logcolor import ColoredFormatter

		# Get the root logger.
		logger = logging.getLogger()

		# Create a new handler that writes to stderr, and attach the
		# color formatter to it.
		stream = logging.StreamHandler()
		stream.setFormatter(ColorFormatter(
			# $COLOR triggers the start of a colored sequence
			# $REST clears the coloring
			fmt='$COLOR[{levelname} {asctime} '
				'{name}:{lineno}]$RESET {message}',
			style='{'
		))
		logger.addHandler(stream)
		logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
		```

		A convenience function is supplied to make this easy:
		```python
		import logcolor

		logcolor.basicConfig(
			level=(logging.DEBUG if args.verbose else logging.INFO),
			format='$COLOR[{asctime} {name}:{lineno}]$RESET {message}',
			style='{'
		)
		```
	"""

	###########################################################################
	def format(self, record):
		""" Formats the log record using colors.
		"""
		# Get the message using the default formatter
		message = super().format(record)
		# Get the color corresponding to the log-level.
		levelname = record.levelname
		color = COLOR_SEQ.format(30 + COLORS[levelname])
		# Do the replacements to trigger colored outputs
		message = message.replace("$RESET", RESET_SEQ) \
			.replace("$BOLD", BOLD_SEQ).replace("$COLOR", color)
		# Return the result, and terminate it in a reset just to be safe.
		return message + RESET_SEQ

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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

import os
import json
import signal
import atexit
import time
import sys
import argparse
import logging

from . import __version__, __homepage__
from .utils import logcolor
from . import Kurfile
from .engine import JinjaEngine

# get logger a name for display
logger = logging.getLogger(__name__)

###############################################################################
def parse_kurfile(filename, engine):
	""" Parses a Kurfile.

		# Arguments

		filename: str. The path to the Kurfile to load.

		# Return value

		Kurfile instance
	"""
	spec = Kurfile(filename, engine)
	spec.parse()
	return spec

###############################################################################
def dump(args):
	""" Dumps the Kurfile to stdout as a JSON blob.
	"""
	logger.info("execute dump(args) ....")
	spec = parse_kurfile(args.kurfile, args.engine)
	# import json; json.dumps?
	# todo: remove one of `evaluate` and 'evaluation', and etc
	print(json.dumps(spec.data, sort_keys=False, indent=4))

###############################################################################
def train(args):
	""" Trains a model.
	"""
	spec = parse_kurfile(args.kurfile, args.engine)
	func = spec.get_training_function()
	func(step=args.step)

###############################################################################
def test(args):
	""" Tests a model.
	"""

	spec = parse_kurfile(args.kurfile, args.engine)
	func = spec.get_testing_function()
	func(step=args.step)

###############################################################################
def evaluate(args):
	""" Evaluates a model.
	"""
	spec = parse_kurfile(args.kurfile, args.engine)
	func = spec.get_evaluation_function()
	func(step=args.step)

###############################################################################
def build(args):
	""" Builds a model.
	"""
	
	logger.info("Executing build(args) ...  ")

	spec = parse_kurfile(args.kurfile, args.engine)


	# if build compile option is set to auto
	if args.compile == 'auto':
		# create an empty list
		result = []
		# if train, test, evalute sections are available in kurfile,
		for section in ('train', 'test', 'evaluate'):
			if section in spec.data:
				# then store the sections inside result list
				result.append((section, 'data' in spec.data[section]))

		# if no section above is available in the kurfile
		if not result:
			# display message
			logger.info('Trying to build a bare model.')
			# set build complile option to be none
			args.compile = 'none'

		# if those sections are available in kurfile, then
		else:
			logger.info("Hack! what is inside result: ----------")
			logger.info("result without sorted: %s", result)
			args.compile, has_data = sorted(result, key=lambda x: not x[1])[0]
			logger.info("result after sorted: %s", sorted(result))

			logger.info('Trying to build a "%s" model.', args.compile)
			if not has_data:
				logger.info('There is not data defined for this model, '
					'though, so we will be running as if --bare was '
					'specified.')
	elif args.compile == 'none':
		logger.info('Trying to build a bare model.')
	else:
		logger.info('Trying to build a "%s" model.', args.compile)

	if args.bare or args.compile == 'none':
		provider = None
	else:
		provider = spec.get_provider(args.compile)

	spec.get_model(provider)

	if args.compile == 'none':
		return
	elif args.compile == 'train':
		target = spec.get_trainer(with_optimizer=True)
	elif args.compile == 'test':
		target = spec.get_trainer(with_optimizer=False)
	elif args.compile == 'evaluate':
		target = spec.get_evaluator()
	else:
		logger.error('Unhandled compilation target: %s. This is a bug.',
			args.compile)
		return 1

	target.compile()

###############################################################################
def prepare_data(args):
	""" Prepares a model's data provider.
	"""
	spec = parse_kurfile(args.kurfile, args.engine)

	if args.target == 'auto':
		result = None
		for section in ('train', 'validate', 'test', 'evaluate'):
			if section in spec.data and 'data' in spec.data[section]:
				result = section
				break
		if result is None:
			raise ValueError('No data sections were found in the Kurfile.')
		args.target = result

	logger.info('Preparing data sources for: %s', args.target)

	provider = spec.get_provider(args.target)

	if args.assemble:

		spec.get_model(provider)

		if args.target == 'train':
			target = spec.get_trainer(with_optimizer=True)
		elif args.target == 'test':
			target = spec.get_trainer(with_optimizer=False)
		elif args.target == 'evaluate':
			target = spec.get_evaluator()
		else:
			logger.error('Unhandled assembly target: %s. This is a bug.',
				args.target)
			return 1

		target.compile(assemble_only=True)

	batch = None
	for batch in provider:
		break
	if batch is None:
		logger.error('No batches were produced.')
		return 1

	num_entries = None
	keys = sorted(batch.keys())
	num_entries = len(batch[keys[0]])
	for entry in range(num_entries):
		print('Entry {}/{}:'.format(entry+1, num_entries))
		for k in keys:
			print('  {}: {}'.format(k, batch[k][entry]))

	if num_entries is None:
		logger.error('No data sources was produced.')
		return 1

###############################################################################
def version(args):							# pylint: disable=unused-argument
	""" Prints the Kur version and exits.
	"""
	logger.info("version(args) is running ...")
	print('Kur, by Deepgram -- deep learning made easy')
	print('Version: {}'.format(__version__))
	print('Homepage: {}'.format(__homepage__))

###############################################################################
def do_monitor(args):
	""" Handle "monitor" mode.
	"""

	# If we aren't running in monitor mode, then we are done.
	if not args.monitor:
		return

	# This is the main retry loop.
	while True:
		# Fork the process.
		logger.info('Forking child process.')
		pid = os.fork()

		# If we are the child, leave this function and work.
		if pid == 0:
			logger.info('We are a newly spawned child process.')
			return

		logger.info('Child process spawned: %d', pid)

		# Wait for the child to die. If we die first, kill the child.
		atexit.register(kill_process, pid)
		try:
			_, exit_status = os.waitpid(pid, 0)
		except KeyboardInterrupt:
			break
		atexit.unregister(kill_process)

		# Process the exit code.
		signal_number = exit_status & 0xFF
		exit_code = (exit_status >> 8) & 0xFF
		core_dump = bool(0x80 & signal_number)

		if signal_number == 0:
			logger.info('Child process exited with exit code: %d.', exit_code)
		else:
			logger.info('Child process exited with signal %d (core dump: %s).',
				signal_number, core_dump)

		retry = False
		if os.WIFSIGNALED(exit_status):
			if os.WTERMSIG(exit_status) == signal.SIGSEGV:
				logger.error('Child process seg faulted.')
				retry = True

		if not retry:
			break

	sys.exit(0)

###############################################################################
def kill_process(pid):
	""" Kills a child process by PID.
	"""

	# Maximum time we wait (in seconds) before we send SIGKILL.
	max_timeout = 60

	# Terminate child process
	logger.info('Sending Ctrl+C to the child process %d', pid)
	os.kill(pid, signal.SIGINT)

	start = time.time()
	while True:
		now = time.time()

		# Check the result.
		result = os.waitpid(pid, os.WNOHANG)
		if result != (0, 0):
			# The child process is dead.
			break

		# Check the timeout.
		if now - start > max_timeout:
			# We've waited too long.
			os.kill(pid, signal.SIGKILL)
			break

		# Keep waiting.
		logger.debug('Waiting patiently...')
		time.sleep(0.5)

###############################################################################
def parse_args():
	""" Constructs an argument parser and returns the parsed arguments.
	"""
	parser = argparse.ArgumentParser(
		description='Descriptive deep learning')
	parser.add_argument('--no-color', action='store_true',
		help='Disable colorful logging.')
	parser.add_argument('-v', '--verbose', default=0, action='count',
		help='Increase verbosity. Can be specified twice for debug-level '
			'output.')
	parser.add_argument('--monitor', action='store_true',
		help='Run Kur in monitor mode, which tries to recover from critical '
			'errors, like segmentation faults.')
	parser.add_argument('--version', action='store_true',
		help='Display version and exit.')

	subparsers = parser.add_subparsers(dest='cmd', help='Sub-command help.')

	subparser = subparsers.add_parser('train', help='Trains a model.')
	subparser.add_argument('--step', action='store_true',
		help='Interactive debug; prompt user before submitting each batch.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	subparser.set_defaults(func=train)

	subparser = subparsers.add_parser('test', help='Tests a model.')
	subparser.add_argument('--step', action='store_true',
		help='Interactive debug; prompt user before submitting each batch.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	subparser.set_defaults(func=test)

	subparser = subparsers.add_parser('evaluate', help='Evaluates a model.')
	subparser.add_argument('--step', action='store_true',
		help='Interactive debug; prompt user before submitting each batch.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	subparser.set_defaults(func=evaluate)

	subparser = subparsers.add_parser('build',
		help='Tries to build a model. This is useful for debugging a model.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	subparser.add_argument('-c', '--compile',
		choices=['none', 'train', 'test', 'evaluate', 'auto'], default='auto',
		help='Try to compile the specified variation of the model. If '
			'--compile=none, then it only tries to assemble the model, not '
			'compile anything. --compile=none implies --bare')
	subparser.add_argument('-b', '--bare', action='store_true',
		help='Do not attempt to load the data providers. In order for your '
			'model to build correctly with this option, you will need to '
			'specify shapes for all of your inputs.')
	subparser.set_defaults(func=build)

	subparser = subparsers.add_parser('dump',
		help='Dumps the Kurfile out as a JSON blob. Useful for debugging.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	subparser.set_defaults(func=dump)

	subparser = subparsers.add_parser('data',
		help='Does not actually compile anything, but only prints out a '
			'single batch of data. This is useful for debugging data sources.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	subparser.add_argument('-t', '--target',
		choices=['train', 'validate', 'test', 'evaluate', 'auto'],
		default='auto', help='Try to produce data corresponding to a specific '
			'variation of the model.')
	subparser.add_argument('--assemble', action='store_true', help='Also '
		'begin assembling the model to pull in compile-time, auxiliary data '
		'sources.')
	subparser.set_defaults(func=prepare_data)

	return parser.parse_args()

###############################################################################
def main():
	""" Entry point for the Kur command-line script.
	"""


	# logging is introduced after this line of code, so
	args = parse_args()



	# set log level for display level (warning, info and debug levels)
	loglevel = {
		0 : logging.WARNING,
		1 : logging.INFO,
		2 : logging.DEBUG
	}


	# A number of optional keyword arguments may be specified, which can alter
	# the default behaviour.
	# to log we must first finish configure
	config = logging.basicConfig if args.no_color else logcolor.basicConfig
	config(

		# choose log level based on value of args.verbose
		# default = 0, `-v` = 1, `-vv` = 2
		level=loglevel.get(args.verbose, logging.DEBUG),

		# set color
		format='{color}[%(levelname)s %(asctime)s %(name)s:%(lineno)s]{reset} '
			'%(message)s'.format(
				color='' if args.no_color else '$COLOR',
				reset='' if args.no_color else '$RESET'
			)
	)


	logger.info("Step1: get console inputs or args into py-script")
	logger.info("args = parse_args(): %s", args)


	# in ipython, try:
	# import logging; logging.captureWarnings?
	logging.captureWarnings(True)



	logger.info("Step2: if args will, do_monitor(args) will run next")
	# do monitor here, but why?
	do_monitor(args)


	logger.info("Step3: if args.version True, kur version will show up")
	# if console receive `--version`, then args.version == True
	if args.version:
		# set args.func = version, version is a function defined above
		args.func = version




	elif not hasattr(args, 'func'):
		logger.info("Step4: if no args.func is given, print out the following")
		## import sys; sys.stderr?
		print('Nothing to do!', file=sys.stderr)
		print('For usage information, try: kur --help', file=sys.stderr)
		print('Or visit our homepage: {}'.format(__homepage__))
		## import sys; sys.exit?
		# exit as failure?
		sys.exit(1)

	# import kur; from kur.engine import JinjaEngine; JinjaEngine.__dict__
	# Creates a new Jinja2 templating engine.
	logger.info("Step5: create a JinjiaEngine object")
	engine = JinjaEngine()

	# setattr(x, 'y', v) is equivalent to ``x.y = v''
	# set args.engine to be engine defined above
	logger.info("Step6: set args.engine to be a real engine object")
	setattr(args, 'engine', engine)


	### This line of code is crucial: it runs all functions as user want expressed as inputs from console
	# comment it out, nothing will actually get done
	# with this line of code, system run the functions and exit with success.
	logger.info("Step7: run relevant funcions with args: it can be version(), train(), evaluate(), test(), dump(), build(), prepare_data(), then exit the program as success")
	sys.exit(args.func(args) or 0)

###############################################################################
if __name__ == '__main__':
	main()

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

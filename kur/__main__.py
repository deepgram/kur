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

import sys
import argparse
import logging
from . import __version__, __homepage__
from .utils import logcolor
from . import Kurfile
from .engine import JinjaEngine

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
def train(args):
	""" Trains a model.
	"""
	spec = parse_kurfile(args.kurfile, args.engine)
	func = spec.get_training_function()
	func()

###############################################################################
def test(args):
	""" Tests a model.
	"""
	spec = parse_kurfile(args.kurfile, args.engine)
	func = spec.get_testing_function()
	func()

###############################################################################
def evaluate(args):
	""" Evaluates a model.
	"""
	spec = parse_kurfile(args.kurfile, args.engine)
	func = spec.get_evaluation_function()
	func()

###############################################################################
def build(args):
	""" Builds a model.
	"""
	spec = parse_kurfile(args.kurfile, args.engine)

	if args.compile == 'auto':
		result = []
		for section in ('train', 'test', 'evaluate'):
			if section in spec.data:
				result.append((section, 'data' in spec.data[section]))
		if not result:
			logger.info('Trying to build a bare model.')
			args.compile = 'none'
		else:
			args.compile, has_data = sorted(result, key=lambda x: not x[1])[0]
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
	print('Kur, by Deepgram -- deep learning made easy')
	print('Version: {}'.format(__version__))
	print('Homepage: {}'.format(__homepage__))

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
	parser.add_argument('--version', action='store_true',
		help='Display version and exit.')

	subparsers = parser.add_subparsers(dest='cmd', help='Sub-command help.')

	subparser = subparsers.add_parser('train', help='Trains a model.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	subparser.set_defaults(func=train)

	subparser = subparsers.add_parser('test', help='Tests a model.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	subparser.set_defaults(func=test)

	subparser = subparsers.add_parser('evaluate', help='Evaluates a model.')
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
	args = parse_args()

	if args.version:
		args.func = version
	elif not hasattr(args, 'func'):
		print('Nothing to do!', file=sys.stderr)
		print('For usage information, try: kur --help', file=sys.stderr)
		print('Or visit our homepage: {}'.format(__homepage__))
		sys.exit(1)

	loglevel = {
		0 : logging.WARNING,
		1 : logging.INFO,
		2 : logging.DEBUG
	}
	config = logging.basicConfig if args.no_color else logcolor.basicConfig
	config(
		level=loglevel.get(args.verbose, logging.DEBUG),
		format='{color}[%(levelname)s %(asctime)s %(name)s:%(lineno)s]{reset} '
			'%(message)s'.format(
				color='' if args.no_color else '$COLOR',
				reset='' if args.no_color else '$RESET'
			)
	)
	logging.captureWarnings(True)

	engine = JinjaEngine()
	setattr(args, 'engine', engine)

	sys.exit(args.func(args) or 0)

###############################################################################
if __name__ == '__main__':
	main()

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

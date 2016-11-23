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
from .utils import logcolor
from .parsing import Specification
from .engine import JinjaEngine

################################################################################
def parse_specification(filename, engine):
	""" Parses a specification file.

		# Arguments

		filename: str. The path to the specification file to load.

		# Return value

		Specification instance
	"""
	spec = Specification(filename, engine)
	spec.parse()
	return spec

################################################################################
def train(args):
	""" Trains a model.
	"""
	spec = parse_specification(args.specification, args.engine)
	func = spec.get_training_function()
	func()

################################################################################
def test(args):
	""" Tests a model.
	"""
	spec = parse_specification(args.specification, args.engine)
	func = spec.get_testing_function()
	func()

################################################################################
def evaluate(args):
	""" Evaluates a model.
	"""
	spec = parse_specification(args.specification, args.engine)
	func = spec.get_evaluation_function()
	func()

################################################################################
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

	subparsers = parser.add_subparsers(dest='cmd', help='Sub-command help.')
	subparsers.required = True

	subparser = subparsers.add_parser('train', help='Trains a model.')
	subparser.add_argument('specification', nargs='?',
		help='The specification file to use.')
	subparser.set_defaults(func=train)

	subparser = subparsers.add_parser('test', help='Tests a model.')
	subparser.add_argument('specification', nargs='?',
		help='The specification file to use.')
	subparser.set_defaults(func=test)

	subparser = subparsers.add_parser('evaluate', help='Evaluates a model.')
	subparser.add_argument('specification', nargs='?',
		help='The specification file to use.')
	subparser.set_defaults(func=evaluate)

	return parser.parse_args()

################################################################################
def main():
	""" Entry point for the Kur command-line script.
	"""
	args = parse_args()

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

################################################################################
if __name__ == '__main__':
	main()

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

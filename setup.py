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

###############################################################################
from __future__ import print_function
import sys

###############################################################################
def error_message(msg):
	""" Prints an error message and exits.
	"""
	line_width = 60
	format_spec = '{{: ^{width}}}'.format(width=line_width)
	lines = [
		'', '',
		'='*line_width, '',
		'ERROR', '',
		msg, ''
		'See our troubleshooting page to get started:', '',
		'https://kur.deepgram.com/troubleshooting.html#installation', '',
		'='*line_width, '',
		"Uh, oh. There was an error. Look up there ^^^^ and you'll be",
		'training awesome models in no time!'
	]
	for line in lines:
		print(format_spec.format(line), file=sys.stderr)
	sys.exit(1)

###############################################################################
if sys.version_info < (3, 4):
	error_message('Kur requires Python 3.4 or later.')

###############################################################################
# pylint: disable=wrong-import-position
import os
from setuptools import setup, find_packages
# pylint: enable=wrong-import-position

################################################################################
def readme():
	""" Return the README text.
	"""
	with open('README.rst', 'rb') as fh:
		result = fh.read()

	result = result.decode('utf-8')

	token = '.. package_readme_ends_here'
	mark = result.find(token)
	if mark >= 0:
		result = result[:mark]

	token = '.. package_readme_starts_here'
	mark = result.find(token)
	if mark >= 0:
		result = result[mark+len(token):]

	chunks = []
	skip = False
	for chunk in result.split('\n\n'):
		if not chunk:
			pass
		elif chunk.strip().startswith('.. package_readme_ignore'):
			skip = True
		elif skip:
			skip = False
		else:
			chunks.append(chunk)

	result = '\n\n'.join(chunks)

	return result

################################################################################
def get_version():
	""" Gets the current version of the package.
	"""
	version_py = os.path.join(os.path.dirname(__file__), 'kur', 'version.py')
	with open(version_py, 'r') as fh:
		for line in fh:
			if line.startswith('__version__'):
				return line.split('=')[-1].strip().replace('"', '')
	raise ValueError('Failed to parse version from: {}'.format(version_py))

################################################################################
setup(
	# Package information
	name='kur',
	version=get_version(),
	description='Descriptive deep learning',
	long_description=readme(),
	keywords='deep learning',
	classifiers=[
	],

	# Author information
	url='https://github.com/deepgram/kur',
	author='Adam Sypniewski',
	author_email='adam@deepgram.com',
	license='Apache Software License '
		'(http://www.apache.org/licenses/LICENSE-2.0)',

	# What is packaged here.
	packages=find_packages(),

	# What to include.
	package_data={
		'': ['*.txt', '*.rst', '*.md']
	},

	# Dependencies
	install_requires=[
		'pyyaml>=3.12',
		'jinja2>=2.8',
		'numpy>=1.11.2',
		'tqdm>=4.10.0',

		# Keras - the default backend (with Theano)
		'keras>=1.1.2',
		'theano>=0.8.2',

		'scipy>=0.18.1',
		'python-magic>=0.4.12',
		'pydub>=0.16.6',
		'python_speech_features>=0.4',
		'matplotlib>=1.5.3'
	],
	dependency_links=[
	],

	# Testing
	test_suite='tests',
	tests_require=[
		'pytest',
		'tensorflow'
	],
	setup_requires=['pytest-runner'],

	entry_points={
		'console_scripts' : ['kur=kur.__main__:main']
	},

	zip_safe=False
)

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

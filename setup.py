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
import subprocess
import warnings
from setuptools import setup, find_packages

################################################################################
def readme():
	""" Return the README text.
	"""
	with open('README.rst') as fh:
		return fh.read()

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
		'pyyaml',
		'jinja2',
		'numpy',
		'requests',
		'tqdm',

		# Keras - the default backend (with Theano)
		'keras',
		'theano'
	],
	dependency_links=[
	],

	# Testing
	test_suite='tests',
	tests_require=['pytest', 'tensorflow'],
	setup_requires=['pytest-runner'],

	entry_points={
		'console_scripts' : ['kur=kur.__main__:main']
	},

	zip_safe=False
)

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

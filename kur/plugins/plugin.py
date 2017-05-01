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
import sys
import importlib.util
import shutil
import subprocess
import logging
from functools import wraps

from ..utils import can_import

logger = logging.getLogger(__name__)

###############################################################################
class Plugin:
	""" Base class for all Kur plugins.
	"""

	PLUGINS = set()
	PLUGIN_DIR = '~/.kur/plugins'
	PLUGIN_FILE = 'enabled'

	###########################################################################
	@classmethod
	def exists(cls, name):
		""" Returns True if a plugin exists.
		"""
		return can_import(name)

	###########################################################################
	@classmethod
	def install(cls, name):
		""" Installs a plugin from PyPI.
		"""
		if cls.exists(name):
			return True

		while True:
			response = input('Plugin "{}" does not exist. Can we try to '
				'install it? [Y/n]'.format(name)).lower()
			if not response or response in {'y', 'yes'}:
				break
			elif response in {'n', 'no'}:
				return False
		pip = shutil.which('pip')
		if not pip:
			logger.error('The built-in plugin installer uses "pip" to install '
				'plugins, but "pip" does not seem to be available. Instead, '
				'you should install the plugin manually and then enable it.')
			return False
		proc = subprocess.Popen(
			[pip, 'install', name],
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			universal_newlines=True
		)
		_, stderr = proc.communicate()
		if proc.returncode:
			logger.error('Failed to install the plugin. Reason: %s',
				stderr.replace('\n', ' '))
			return False

		return True

	###########################################################################
	@classmethod
	def inject_parser(cls, name, injector):
		""" Injects a new parser section into the argument parser.
		"""
		if name in cls.PLUGINS:
			return

		import kur.__main__ as main
		old_builder = main.build_parser

		@wraps(old_builder)
		def new_builder():
			""" The new parser.
			"""
			parser, subparsers = old_builder()
			result = injector(parser, subparsers)
			if result is None:
				return parser, subparsers
			return result

		main.build_parser = new_builder
		cls.PLUGINS.add(name)

	###########################################################################
	@classmethod
	def get_plugin_file(cls):
		""" Returns the path to the enabled plugins file.
		"""
		return os.path.expanduser(
			os.path.expandvars(
				os.path.join(cls.PLUGIN_DIR, cls.PLUGIN_FILE)
			)
		)

	###########################################################################
	@classmethod
	def get_enabled_plugins(cls):
		""" Returns a list of Plugin handlers for enabled plugins.
		"""
		plugins = []
		if not os.path.isfile(cls.get_plugin_file()):
			return plugins
		with open(cls.get_plugin_file()) as fh:
			for line in fh:
				line = line.strip()
				if not line:
					continue
				try:
					plugin = Plugin(line)
				except ValueError:
					logger.warning('A plugin was marked as enabled, but does '
						'not seem to be installed: %s', line)
				else:
					plugins.append(plugin)
		return plugins

	###########################################################################
	def __init__(self, name):
		""" Creates a new plugin handler.
		"""
		if not self.exists(name):
			raise ValueError('No such plugin exists: {}'.format(name))

		self.name = name

	###########################################################################
	def __str__(self):
		return self.name

	###########################################################################
	def __repr__(self):
		return 'Plugin("{}")'.format(self.name)

	###########################################################################
	@property
	def enabled(self):
		""" Returns True if the plugin is enabled, and False otherwise.
		"""
		if not os.path.isfile(self.get_plugin_file()):
			return False
		with open(self.get_plugin_file()) as fh:
			for line in fh:
				line = line.strip()
				if not line:
					continue
				if line == self.name:
					return True
		return False

	###########################################################################
	@enabled.setter
	def enabled(self, value):
		""" Sets the enabled state of the plugin.
		"""
		value = bool(value)
		if self.enabled == value:
			return
		if value:
			# Make sure we are loaded. If we cannot load, we should not enable
			# ourself, because this would break Kur.
			self.load()
			if not os.path.isfile(self.get_plugin_file()):
				os.makedirs(
					os.path.dirname(self.get_plugin_file()), exist_ok=True
				)
			with open(self.get_plugin_file(), 'a') as fh:
				fh.write(self.name + '\n')
		else:
			if os.path.isfile(self.get_plugin_file()):
				return
			result = []
			with open(self.get_plugin_file(), 'w+') as fh:
				for line in fh:
					line = line.strip()
					if not line:
						continue
					if line != self.name:
						result.append(line)
				fh.seek(0)
				fh.write('\n'.join(result) + '\n')

	###########################################################################
	@property
	def loaded(self):
		""" Checks if the plugin has been loaded.
		"""
		return self.name in sys.modules

	###########################################################################
	def load(self):
		""" Loads the plugin.
		"""
		if self.loaded:
			return

		spec = importlib.util.find_spec(self.name)
		if spec is None:
			raise ValueError('Cannot find spec for module: {}. Are you sure '
				'it is installed? If it is, is its virtual environment active?'
				.format(self.name))

		module = importlib.util.module_from_spec(spec)

		# This serves two purposes:
		# 1. It allows the module to be referenced by name later on (although
		#    this probably doesn't really matter for plugins).
		# 2. It causes the parent module to appear loaded, allowing the plugin
		#    to follow standard package-relative imports, thereby avoiding this
		#    annoying error:
		#        SystemError: Parent module 'X' not loaded, cannot perform
		#                     relative import
		sys.modules[self.name] = module

		# Load the module
		spec.loader.exec_module(module)

		if not hasattr(module, '__kur__') or not module.__kur__:
			raise ValueError('This is not a valid Kur plugin.')

		# Run its setup call.
		if not hasattr(module, 'setup'):
			raise ValueError('This is not a valid Kur plugin.')
		module.setup()

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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
import copy
import socket
import warnings
import logging
from .engine import ScopeStack, PassthroughEngine
from .reader import Reader
from .containers import Container, ParsingError
from .model import Model, Executor, EvaluationHook, OutputHook, TrainingHook
from .backend import Backend
from .optimizer import Optimizer, Adam
from .loss import Loss
from .providers import Provider, BatchProvider, ShuffleProvider
from .supplier import Supplier
from .utils import mergetools
from .loggers import Logger

logger = logging.getLogger(__name__)

###############################################################################
class Kurfile:
	""" Class for loading and parsing Kurfiles.
	"""

	DEFAULT_OPTIMIZER = Adam
	DEFAULT_PROVIDER = BatchProvider

	###########################################################################
	def __init__(self, source, engine=None):
		""" Creates a new Kurfile.

			# Arguments

			source: str or dict. If it is a string, it is interpretted as a
				filename to an on-disk Kurfile. Otherwise, it is
				interpretted as an already-loaded, but unparsed, Kurfile.
			engine: Engine instance. The templating engine to use in parsing.
				If None, a Passthrough engine is instantiated.
		"""
		engine = engine or PassthroughEngine()
		if isinstance(source, str):
			filename = os.path.expanduser(os.path.expandvars(source))
			if not os.path.isfile(filename):
				raise IOError('No such file found: {}. Path was expanded to: '
					'{}.'.format(source, filename))
			self.filename = filename
			self.data = self.parse_source(
				engine,
				source=filename,
				context=None
			)
		else:
			self.filename = None
			self.data = dict(source)

		self.containers = None
		self.model = None
		self.backend = None
		self.engine = engine

	###########################################################################
	def parse(self):
		""" Parses the Kurfile.
		"""

		logger.info('Parsing Kurfile...')

		# These are reserved section names.
		# The first name in each tuple is the one that we should rely on
		# internally when we reference `self.data` in Kurfile.
		builtin = {
			'settings' : ('settings', ),
			'train' : ('train', 'training'),
			'validate' : ('validate', 'validation'),
			'test' : ('test', 'testing'),
			'evaluate' : ('evaluate', 'evaluation'),
			'model' : ('model', ),
			'loss' : ('loss', )
		}

		# The scope stack.
		stack = []

		# Add default entries.
		stack.append({
			'filename' : self.filename,
			'raw' : copy.deepcopy(self.data),
			'parsed' : self.data,
			'host' : socket.gethostname()
		})

		# Parse the settings (backends, globals, hyperparameters, ...)
		self._parse_section(
			self.engine, builtin['settings'], stack, include_key=False)

		# Parse all of the "action" sections.
		self._parse_section(
			self.engine, builtin['train'], stack, include_key=True)
		self._parse_section(
			self.engine, builtin['validate'], stack, include_key=True)
		self._parse_section(
			self.engine, builtin['test'], stack, include_key=True)
		self._parse_section(
			self.engine, builtin['evaluate'], stack, include_key=True)

		# Parse the model.
		self.containers = self._parse_model(
			self.engine, builtin['model'], stack, required=True)

		# Parse the loss function.
		self._parse_section(
			self.engine, builtin['loss'], stack, include_key=True)

		# Check for unused keys.
		for key in self.data.keys():
			for section in builtin.values():
				if key in section:
					break
			else:
				warnings.warn('Unexpected section in Kurfile: "{}". '
					'This section will be ignored.', SyntaxWarning)

	###########################################################################
	def get_model(self, provider=None):
		""" Returns the parsed Model instance.
		"""
		if self.model is None:
			if self.containers is None:
				raise ValueError('No such model available.')
			self.model = Model(
				backend=self.get_backend(),
				containers=self.containers
			)
			self.model.parse(self.engine)
			self.model.register_provider(provider)
			self.model.build()
		return self.model

	###########################################################################
	def get_backend(self):
		""" Creates a new backend from the Kurfile.

			# Return value

			Backend instance

			# Notes

			- If no "global" section exists, or if no "backend" section exists
			  in the "global" section, this will try to instantiate any
			  supported backend installed on the system.
			- This passes `self.data['global']['backend']` to the Backend
			  factory method, if such keys exist; otherwise, it passes None to
			  the factory method.
		"""
		if self.backend is None:
			self.backend = Backend.from_specification(
				(self.data.get('settings') or {}).get('backend')
			)
		return self.backend

	###########################################################################
	def get_seed(self):
		""" Retrieves the global random seed, if any.

			# Return value

			The `settings.seed` value, as an integer, if such a key exists.
			If the key does not exist, returns None.
		"""

		if 'settings' not in self.data or \
				not isinstance(self.data['settings'], dict):
			return None

		seed = self.data['settings'].get('seed')
		if seed is None:
			return None
		elif not isinstance(seed, int):
			raise ValueError('Unknown format for random seed: {}'.format(seed))

		if seed < 0 or seed >= 2**32:
			logger.warning('Random seeds should be unsigned, 32-bit integers. '
				'We are truncating the seed.')
			seed %= 2**32

		return seed

	###########################################################################
	def get_provider(self, section):
		""" Creates the provider corresponding to a part of the Kurfile.

			# Arguments

			section: str. The name of the section to load the provider for.

			# Return value

			If the section exists and has at least the "data" or "provider"
			section defined, then this returns a Provider instance. Otherwise,
			returns None.
		"""
		if section in self.data:
			section = self.data[section]
			if not any(k in section for k in ('data', 'provider')):
				return None
		else:
			return None

		supplier_list = section.get('data') or []
		if not isinstance(supplier_list, (list, tuple)):
			raise ValueError('"data" section should be a list.')
		suppliers = [
			Supplier.from_specification(x, kurfile=self) for x in supplier_list
		]

		provider_spec = dict(section.get('provider') or {})
		if 'name' in provider_spec:
			provider = Provider.get_provider_by_name(provider_spec.pop('name'))
		else:
			provider = Kurfile.DEFAULT_PROVIDER

		return provider(
			sources=Supplier.merge_suppliers(suppliers),
			**provider_spec
		)

	###########################################################################
	def get_training_function(self):
		""" Returns a function that will train the model.
		"""
		if 'train' not in self.data:
			raise ValueError('Cannot construct training function. There is a '
				'missing "train" section.')

		if 'log' in self.data['train']:
			log = Logger.from_specification(self.data['train']['log'])
		else:
			log = None

		epochs = self.data['train'].get('epochs')

		provider = self.get_provider('train')

		training_hooks = self.data['train'].get('hooks', [])
		if not isinstance(training_hooks, (list, tuple)):
			raise ValueError('"hooks" (in the "train" section) should '
				'be a list of hook specifications.')
		training_hooks = [TrainingHook.from_specification(spec) \
			for spec in training_hooks]

		if 'validate' in self.data:
			validation = self.get_provider('validate')
			validation_weights = self.data['validate'].get('weights')
			if validation_weights is None:
				best_valid = None
			elif isinstance(validation_weights, str):
				best_valid = validation_weights
			elif isinstance(validation_weights, dict):
				best_valid = validation_weights.get('best')
			else:
				raise ValueError('Unknown type for validation weights: {}'
					.format(validation_weights))

			validation_hooks = self.data['validate'].get('hooks', [])
			if not isinstance(validation_hooks, (list, tuple)):
				raise ValueError('"hooks" (in the "validate" section) should '
					'be a list of hook specifications.')
			validation_hooks = [EvaluationHook.from_specification(spec) \
				for spec in validation_hooks]
		else:
			validation = None
			best_valid = None
			validation_hooks = None

		train_weights = self.data['train'].get('weights')
		if train_weights is None:
			initial_weights = best_train = last_weights = None
			deprecated_checkpoint = None
		elif isinstance(train_weights, str):
			initial_weights = train_weights
			best_train = train_weights if best_valid is None else None
			last_weights = None
			initial_must_exist = False
			deprecated_checkpoint = None
		elif isinstance(train_weights, dict):
			initial_weights = train_weights.get('initial')
			best_train = train_weights.get('best')
			last_weights = train_weights.get('last')
			initial_must_exist = train_weights.get('must_exist', False)
			deprecated_checkpoint = train_weights.get('checkpoint')
		else:
			raise ValueError('Unknown weight specification for training: {}'
				.format(train_weights))

		checkpoint = self.data['train'].get('checkpoint')

		if deprecated_checkpoint is not None:
			warnings.warn('"checkpoint" belongs under "train", not under '
				'"weights".', DeprecationWarning)
			if checkpoint is None:
				checkpoint = deprecated_checkpoint
			else:
				logger.warning('The currently-accepted "checkpoint" will be '
					'used over the deprecated "checkpoint".')

		expand = lambda x: os.path.expanduser(os.path.expandvars(x))
		initial_weights, best_train, best_valid, last_weights = [
			expand(x) if x is not None else x for x in
				(initial_weights, best_train, best_valid, last_weights)
		]

		model = self.get_model(provider)
		trainer = self.get_trainer()

		def func(**kwargs):
			""" Trains a model from a pre-packaged specification file.
			"""
			if initial_weights is not None:
				if os.path.exists(initial_weights):
					model.restore(initial_weights)
				elif initial_must_exist:
					logger.error('Configuration indicates that the weight '
						'file must exist, but the weight file was not found.')
					raise ValueError('Missing initial weight file. If you '
						'want to proceed anyway, set "must_exist" to "no" '
						'under the training "weights" section.')
				else:
					if log is not None and \
						(log.get_number_of_epochs() or 0) > 0:
						logger.warning('The initial weights are missing: %s. '
							'However, the logs suggest that this model has '
							'been trained previously. It is possible that the '
							'log file is corrupt/out-of-date, or that the '
							'weight files are not supposed to be missing.',
							initial_weights)
					else:
						logger.info('Ignoring missing initial weights: %s. If '
							'this is undesireable, set "must_exist" to "yes" '
							'in the approriate "weights" section.',
							initial_weights)
			defaults = {
				'provider' : provider,
				'validation' : validation,
				'epochs' : epochs,
				'log' : log,
				'best_train' : best_train,
				'best_valid' : best_valid,
				'last_weights' : last_weights,
				'training_hooks' : training_hooks,
				'validation_hooks' : validation_hooks,
				'checkpoint' : checkpoint
			}
			defaults.update(kwargs)
			return trainer.train(**defaults)

		return func

	###########################################################################
	def get_testing_function(self):
		""" Returns a function that will test the model.
		"""
		if 'test' not in self.data:
			raise ValueError('Cannot construct testing function. There is a '
				'missing "test" section.')

		provider = self.get_provider('test')

		# No reason to shuffle things for this.
		if isinstance(provider, ShuffleProvider):
			provider.randomize = False

		weights = self.data['test'].get('weights')
		if weights is None:
			initial_weights = None
		elif isinstance(weights, str):
			initial_weights = weights
		elif isinstance(weights, dict):
			initial_weights = weights.get('initial')
		else:
			raise ValueError('Unknown weight specification for testing: {}'
				.format(weights))

		hooks = self.data['test'].get('hooks', [])
		if not isinstance(hooks, (list, tuple)):
			raise ValueError('"hooks" (in the "test" section) should be a '
			   'list of hook specifications.')
		hooks = [EvaluationHook.from_specification(spec) for spec in hooks]

		expand = lambda x: os.path.expanduser(os.path.expandvars(x))
		if initial_weights is not None:
			initial_weights = expand(initial_weights)

		model = self.get_model(provider)
		trainer = self.get_trainer(with_optimizer=False)

		def func(**kwargs):
			""" Tests a model from a pre-packaged specification file.
			"""
			if initial_weights is not None:
				if os.path.exists(initial_weights):
					model.restore(initial_weights)
				else:
					logger.error('No weight file found: %s. We will just '
						'proceed with default-initialized weights. This will '
						'test that the system works, but the results will be '
						'terrible.', initial_weights)
			else:
				logger.warning('No weight file specified. We will just '
					'proceed with default-initialized weights. This will '
					'test that the system works, but the results will be '
					'terrible.')
			defaults = {
				'provider' : provider,
				'validating' : False,
				'hooks' : hooks
			}
			defaults.update(kwargs)
			return trainer.test(**defaults)

		return func

	###########################################################################
	def get_trainer(self, with_optimizer=True):
		""" Creates a new Trainer from the Kurfile.

			# Return value

			Trainer instance.
		"""
		return Executor(
			model=self.get_model(),
			loss=self.get_loss(),
			optimizer=self.get_optimizer() if with_optimizer else None
		)

	###########################################################################
	def get_optimizer(self):
		""" Creates a new Optimizer from the Kurfile.

			# Return value

			Optimizer instance.
		"""
		if 'train' not in self.data:
			raise ValueError('Cannot construct optimizer. There is a missing '
				'"train" section.')

		spec = dict(self.data['train'].get('optimizer', {}))
		if 'name' in spec:
			optimizer = Optimizer.get_optimizer_by_name(spec.pop('name'))
		else:
			optimizer = Kurfile.DEFAULT_OPTIMIZER

		return optimizer(**spec)

	###########################################################################
	def get_loss(self):
		""" Creates a new Loss from the Kurfile.

			# Return value

			If 'loss' is defined, then this returns a dictionary whose keys are
			output layer names and whose respective values are Loss instances.
			Otherwise, this returns None.
		"""
		if 'loss' not in self.data:
			return None

		spec = self.data['loss']
		if not isinstance(spec, (list, tuple)):
			raise ValueError('"loss" section expects a list of loss '
				'functions.')

		result = {}
		for entry in spec:
			entry = dict(entry)
			target = entry.pop('target', None)
			if target is None:
				raise ValueError('Each loss function in the "loss" list '
					'must have a "target" entry which names the output layer '
					'it is attached to.')

			name = entry.pop('name', None)
			if name is None:
				raise ValueError('Each loss function in the "loss" list '
					'must have a "name" entry which names the loss function '
					'to use for that output.')

			if target in result:
				logger.warning('A loss function for the "%s" output was '
					'already defined. We will replace it by this new one we '
					'just found.', target)
			result[target] = Loss.get_loss_by_name(name)(**entry)

		return result

	###########################################################################
	def get_evaluation_function(self):
		""" Returns a function that will evaluate the model.
		"""
		if 'evaluate' not in self.data:
			raise ValueError('Cannot construct evaluation function. There is '
				'a missing "evaluate" section.')

		provider = self.get_provider('evaluate')

		# No reason to shuffle things for this.
		if isinstance(provider, ShuffleProvider):
			provider.randomize = False

		weights = self.data['evaluate'].get('weights')
		if weights is None:
			initial_weights = None
		elif isinstance(weights, str):
			initial_weights = weights
		elif isinstance(weights, dict):
			initial_weights = weights.get('initial')
		else:
			raise ValueError('Unknown weight specification for evaluation: {}'
				.format(weights))

		destination = self.data['evaluate'].get('destination')
		if isinstance(destination, str):
			destination = OutputHook(path=destination)
		elif isinstance(destination, dict):
			destination = OutputHook(**destination)
		elif destination is not None:
			ValueError('Expected a string or dictionary value for '
				'"destination" in "evaluate". Received: {}'.format(
					destination))

		hooks = self.data['evaluate'].get('hooks', [])
		if not isinstance(hooks, (list, tuple)):
			raise ValueError('"hooks" should be a list of hook '
				'specifications.')
		hooks = [EvaluationHook.from_specification(spec) for spec in hooks]

		expand = lambda x: os.path.expanduser(os.path.expandvars(x))
		if initial_weights is not None:
			initial_weights = expand(initial_weights)

		model = self.get_model(provider)
		evaluator = self.get_evaluator()

		def func(**kwargs):
			""" Evaluates a model from a pre-packaged specification file.
			"""
			if initial_weights is not None:
				if os.path.exists(initial_weights):
					model.restore(initial_weights)
				else:
					logger.error('No weight file found: %s. We will just '
						'proceed with default-initialized weights. This will '
						'test that the system works, but the results will be '
						'terrible.', initial_weights)
			else:
				logger.warning('No weight file specified. We will just '
					'proceed with default-initialized weights. This will '
					'test that the system works, but the results will be '
					'terrible.')
			defaults = {
				'provider' : provider,
				'callback' : None
			}
			defaults.update(kwargs)
			result, truth = evaluator.evaluate(**defaults)
			orig = (result, truth)
			prev = orig
			for hook in hooks:
				new_prev = hook.apply(prev, orig, model)
				prev = (new_prev, prev[1]) \
					if not isinstance(new_prev, tuple) else new_prev
			if destination is not None:
				new_prev = destination.apply(prev, orig, model)
				prev = (new_prev, prev[1]) \
					if not isinstance(new_prev, tuple) else new_prev
			return prev

		return func

	###########################################################################
	def get_evaluator(self):
		""" Creates a new Evaluator from the Kurfile.

			# Return value

			Evaluator instance.
		"""
		return Executor(
			model=self.get_model()
		)

	###########################################################################
	def _parse_model(self, engine, section, stack, *, required=True):
		""" Parses the top-level "model" entry.

			# Arguments

			engine: Engine instance. The templating engine to use in
				evaluation.
			section: str or list of str. The key indicating which top-level
				entry to parse. If a list, the first matching key is used; this
				allows for multiple aliases to the same section.
			stack: list. The list of scope dictionaries to use in evaluation.
			required: bool (default: True). If True, an exception is raised if
				the key is not present in the Kurfile.

			# Return value

			If the model section is found, the list of containers is returned.
			If the model section is not found but is required, an exception is
			raised. If the model section is not found and isn't required, None
			is returned.
		"""
		if isinstance(section, str):
			section = (section, )

		key = None		# Not required, but shuts pylint up.
		for key in section:
			if key in self.data:
				break
		else:
			if required:
				raise ValueError(
					'Missing required section: {}'.format(', '.join(section)))
			else:
				return None

		if not isinstance(self.data[key], (list, tuple)):
			raise ValueError(
				'Section "{}" should contain a list of layers.'.format(key))

		containers = [
			Container.create_container_from_data(entry)
			for entry in self.data[key]
		]

		with ScopeStack(engine, stack):
			for container in containers:
				container.parse(engine)

		return containers

	###########################################################################
	def parse_source(self, engine, *,
		source=None, context=None, loaded=None):
		""" Parses a source, and its includes (recursively), and returns the
			merged sources.
		"""
		def get_id(filename):
			""" Returns a tuple which uniquely identifies a file.

				This function follows symlinks.
			"""
			filename = os.path.expanduser(os.path.expandvars(filename))
			stat = os.stat(filename)
			return (stat.st_dev, stat.st_ino)

		# Process the "loaded" iterable.
		loaded = loaded or set()

		if isinstance(source, str):
			filename = source
			strategy = None
		elif isinstance(source, dict):
			if 'source' not in source:
				raise ParsingError('Error while parsing source file {}. '
					'Missing required "source" key in the include '
					'dictionary: {}.'.format(context or 'top-level', source))
			filename = source.pop('source')
			strategy = source.pop('method', None)
			for k in source:
				warnings.warn('Warning while parsing source file {}. '
					'Ignoring extra key in an "include" dictionary: {}.'
					.format(context or 'top-level', k))
		else:
			raise ParsingError('Error while parsing source file {}. '
				'Expected each "include" to be a string or a dictionary. '
				'Received: {}'.format(context or 'top-level', source))

		logger.info('Parsing source: %s, included by %s.', source,
			context or 'top-level')

		if context:
			filename = os.path.join(os.path.dirname(context), filename)
		expanded = os.path.expanduser(os.path.expandvars(filename))
		if not os.path.isfile(expanded):
			raise IOError('Error while parsing source file {}. No such '
				'source file found: {}. Path was expanded to: {}'.format(
				context or 'top-level', filename, expanded))
		file_id = get_id(expanded)
		if file_id in loaded:
			logger.warning('Skipping an already included source file: %s. '
				'This may have unintended consequences, since the merge '
				'order may result in different final results. You should '
				'refactor your Kurfile to avoid circular includes.',
				expanded)
			return {}
		else:
			loaded.add(file_id)

		try:
			data = Reader.read_file(expanded)
		except:
			logger.exception('Failed to read file: %s. Check your syntax.',
				expanded)
			raise

		new_sources = data.pop('include', [])
		engine.evaluate(new_sources, recursive=True)
		if isinstance(new_sources, str):
			new_sources = [new_sources]
		elif not isinstance(new_sources, (list, tuple)):
			raise ValueError('Error while parsing file: {}. All "include" '
				'sections must be a single source file or a list of source '
				'files. Instead we received: {}'.format(
					expanded, new_sources))

		for x in new_sources:
			data = mergetools.deep_merge(
				self.parse_source(
					engine,
					source=x,
					context=expanded,
					loaded=loaded
				),
				data,
				strategy=strategy
			)

		return data

	###########################################################################
	def _parse_section(self, engine, section, stack, *,
		required=False, include_key=True):
		""" Parses a single top-level entry in the Kurfile.

			# Arguments

			engine: Engine instance. The templating engine to use in
				evaluation.
			section: str or list of str. The key indicating which top-level
				entry to parse. If a list, the first matching key is used; this
				allows for multiple aliases to the same section.
			stack: list. The list of scope dictionaries to use in evaluation.
			required: bool (default: False). If True, an exception is raised if
				the key is not present in the Kurfile.
			include_key: bool (default: False). If True, the parsed section is
				added to the scope stack as a dictionary with a single item,
				whose key is the name of the parsed section, and whose value is
				the evaluated section. If False, then the evaluated section is
				added directly to the scope stack (it must evaluate to a
				dictionary for this to work).

			# Return value

			The parsed section.

			# Note

			- This will replace the values of `self.data` with the evaluated
			  version.
		"""
		if isinstance(section, str):
			section = (section, )

		logger.debug('Parsing Kurfile section: %s', section[0])

		key = None		# Not required, but shuts pylint up.
		for key in section:
			if key in self.data:
				break
		else:
			if required:
				raise ValueError(
					'Missing required section: {}'.format(', '.join(section)))
			else:
				return None

		with ScopeStack(engine, stack):
			evaluated = engine.evaluate(self.data[key], recursive=True)

		if include_key:
			stack.append({key : evaluated})
		else:
			if evaluated is not None:
				if not isinstance(evaluated, dict):
					raise ValueError(
						'Section "{}" should contain key/value pairs.'
						.format(key))
				stack.append(evaluated)

		# Now, just in case this was aliased, let's make sure we keep our
		# naming scheme consistent.
		for key in section:
			self.data[key] = evaluated

		return evaluated

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

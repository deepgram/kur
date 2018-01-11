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

import re
import logging
from collections import OrderedDict

from ..loss import Loss
from ..utils import get_subclasses, CudaContext, CudaError

logger = logging.getLogger(__name__)

###############################################################################
class Backend:
	""" Base class for all tensor operation backends. Examples of backends are
		Keras, Theano, TensorFlow, etc.
	"""

	###########################################################################
	def __init__(self, variant=None, device=None, parallel=None,
		gpu_busy_load=None, skip_check=None):
		""" Create a new backend.

			Part of this call should be to ensure that all the necessary
			modules/libraries are available to use this backend. If this
			backend cannot be used, an exception should be raised. Otherwise,
			any appropriate configuration should take place here.

			# Arguments

			variant: str, list/tuple of str, or None (default: None). The
				variants to label this backend with. This allows for more
				streamlined and flexible modification of backend behavior at
				the layer level. These are simply tags that software components
				can query and modify their behavior in response, without
				requiring an entirely new backend.
		"""
		if gpu_busy_load is None:
			gpu_busy_load = 5.0

		if skip_check is None:
			skip_check = False

		if not self.is_supported():
			logger.warning('Backend claims to not be supported. We will try '
				'to use it anyway.')

		if variant is not None:
			if isinstance(variant, str):
				variant = {variant}
		else:
			variant = set()

		self.variant = set(variant)
		self.variant.discard(None)

		if parallel is not None:
			if not isinstance(parallel, int):
				raise ValueError('If "parallel" is specified, it must be an '
					'integer. We received: {}'.format(parallel))
			if parallel <= 0:
				logger.warning('"parallel" should be > 0, but we received '
					'"%d". We are just ignoring "parallel".', parallel)
				parallel = None

		if device is None:
			logger.debug('No execution device indicated to backend. Checking '
				'available devices...')
			try:
				with CudaContext() as context:
					n_devices = len(context)
			except CudaError:
				logger.debug('Failed to initialize CUDA. Falling back to '
					'CPU.')
				device = 'cpu'
			else:
				if n_devices > 0:
					logger.debug('GPU capabilities detected.')
					device = 'gpu'
				else:
					logger.debug('No GPUs detected.')
					device = 'cpu'

		if device == 'cpu':
			self.devices = []
			if parallel:
				logger.warning('"parallel" execution was requested, but we '
					'not using the GPU. "parallel" will be ignored.')

		elif device.startswith('gpu'):
			if device == 'gpu':
				allowed = None
			else:
				allowed = set()
				regex = re.compile(
					r'gpu(?P<negate>!)?(?P<start>[0-9]+)(?:-(?P<end>[0-9]+))?')
				parts = device.split(',')
				for part in parts:
					match = regex.match(part.strip())
					if not match:
						raise ValueError('Failed to parse device: {}'.format(
							device))
					if match.group('end'):
						indices = set(range(
							int(match.group('start')),
							int(match.group('end'))+1
						))
					else:
						indices = {int(match.group('start'))}
					if match.group('negate') is None:
						allowed |= indices
					else:
						allowed -= indices

			with CudaContext() as context:
				n_devices = len(context)
				if allowed is None:
					allowed = set(range(n_devices))
				if skip_check:
					preferred = list(sorted(allowed))
				else:
					preferred = [
						(device.is_busy(gpu_busy_load), device.index()) \
							for device in context.rank_available()
					]
					preferred = [
						index for busy, index in preferred
						if index in allowed and not busy
					]

			if not preferred:
				logger.error('Failed to find any available GPUs.')
				raise ValueError('Failed to find any available GPUs.')

			if parallel:
				if parallel > len(preferred):
					logger.warning('%d GPU devices were requested through '
						'"parallel", but only %d devices seem '
						'available/unused. We are decreasing "parallel" to '
						'match.', parallel, len(preferred))
					self.devices = preferred
				else:
					self.devices = preferred[:parallel]
			else:
				self.devices = preferred

			logger.info('We are going to use %d GPU devices.',
				len(self.devices))

		else:
			raise ValueError('Invalid device specification: {}. If a '
				'device is explicitly specified, it must be "cpu", "gpu" '
				'or "gpuX" where "X" is an integer.'.format(device))

		logger.info('Creating backend: %s', self.get_name())
		logger.debug('Backend variants: %s',
			'none' if not self.variant else ', '.join(
				str(x) for x in sorted(self.variant)
			)
		)

	###########################################################################
	@property
	def parallel(self):
		""" Returns the number of GPU devices we are using.
		"""
		if self.devices:
			return len(self.devices)
		return 0

	###########################################################################
	def has_variant(self, variant):
		""" Checks if a particular variant is enabled.
		"""
		return variant in self.variant

	###########################################################################
	@classmethod
	def is_supported(cls):
		""" Returns True if this backend can be used.

			# Return value

			If the backend appears to be useable, returns True. Otherwise,
			returns False.

			Note that if this returns False, then the backend should definitely
			not be able to be used. However, just because it returns True
			doesn't mean it will work (e.g., Theano with `force_device=true`,
			but without the requested device installed).
		"""
		raise NotImplementedError

	###########################################################################
	@staticmethod
	def from_specification(spec):
		""" Creates a new backend from the specification.

			# Arguments

			spec: str or dict. The backend specification. If processed from
				a standard job specification, it is everything under the
				`global.backend` section.

			# Return value

			Backend instance

			# Usage

			If no "global" section exists, or if no "backend" section exists in
			the "global" section, this will try to instantiate any supported
			backend installed on the system.

			Instantiate a specific backend with default parameters:
			```yaml
			global:
			  backend: BACKEND
			```
			or like this:
			```yaml
			global:
			  backend:
			    name: BACKEND
			```

			Instantiate a specific backend with additional keywords (the
			keywords must be supported by the Backend implementation):
			```yaml
			global:
			  backend:
			    name: BACKEND
				PARAM1: VALUE1
				PARAM2: VALUE2
				  ...
			```

			Instatiate any backend with additional keywords (the keywords must
			be supported by the Backend implementation):
			```yaml
			global:
			  backend:
				PARAM1: VALUE1
				PARAM2: VALUE2
				...
			```
		"""
		if spec is None:
			all_supported = list(Backend.get_all_backends(supported_only=True))
			if not all_supported:
				raise ValueError('No supported backends available.')
			target = all_supported[0]
			params = {}
		elif isinstance(spec, str):
			target = Backend.get_backend_by_name(spec)
			params = {}
		elif isinstance(spec, dict):
			if 'name' in spec:
				target = Backend.get_backend_by_name(spec.pop('name'))
			else:
				all_supported = list(Backend.get_all_backends(
					supported_only=True))
				if not all_supported:
					raise ValueError('No supported backends available.')
				target = all_supported[0]
			params = spec
		else:
			raise ValueError(
				'Unexpected backend specification: {}'.format(spec))

		logger.debug('Using backend: %s', target.get_name())
		result = target(**params)
		return result

	###########################################################################
	@staticmethod
	def get_backend_by_name(name):
		""" Factory method for creating new backends.

			# Arguments

			name: str. The name of the backend to instantiate. This is given by
				the backend's `get_name()` method.
			kwargs: dict. The arguments to pass to the backend's constructor.

			# Return value

			If the named backend is found, this returns a Backend instance;
			otherwise, a ValueError is raised.
		"""
		name = name.lower()
		for cls in Backend.get_all_backends(supported_only=False):
			if cls.get_name() == name:
				return cls
		raise ValueError('No such backend: {}'.format(name))

	###########################################################################
	@staticmethod
	def get_all_backends(supported_only=False):
		""" Factory method for creating backend which is supported on the host
			platform.

			# Arguments

			kwargs: dict. The arguments to pass to the backend's constructor.

			# Return value

			If the named backend is found, this returns a Backend instance;
			otherwise, a ValueError is raised.
		"""
		for cls in get_subclasses(Backend):
			if supported_only:
				if cls.is_supported():
					yield cls
			else:
				yield cls

	###########################################################################
	@staticmethod
	def get_any_supported_backend():
		""" Finds a backend class that claims to be supported on this system.
		"""
		for cls in Backend.get_all_backends(supported_only=True):
			return cls
		raise ValueError('No supported backends found.')

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Return a lower-case string naming this backend.
		"""
		return cls.__name__.lower()

	###########################################################################
	def save(self, model, filename):
		""" Saves the model weights to the given filename.

			# Arguments

			model: Model instance. The model whose weights should be saved.
			filename: str. The filename to write the weights to.

			# Notes

			The file format is backend-specific. There is no guarantee of
			compatability between different backends.

			# Return value

			None
		"""
		raise NotImplementedError

	###########################################################################
	def restore(self, model, filename):
		""" Load the model weights from the given filename.

			# Arguments

			model: Model instance. The model whose weights should be restored.
			filename: str. The filename to read the weights from.

			# Notes

			The file format is backend-specific. There is no guarantee of
			compatability between different backends.

			# Return value

			None
		"""
		raise NotImplementedError

	###########################################################################
	def create_data(self, model):
		""" Requests a new set of model-specific data to be used during model
			assembly.
		"""
		return None

	###########################################################################
	def connect(self, inputs, target, data):
		""" Applies a tensor operation to a set of input tensors.

			# Arguments

			inputs: list. A list of input tensors.
			target: object. The tensor operation to apply.
			data: object. A backend-specific object, created by
				`create_data()`, which can be used by the backend to maintain
				state while constructing the model.

			# Notes

			The exact underlying types of the `inputs` and `target` tensors
			depend on the backend implementation.

			# Return value

			A new tensor object (backend-specific) resulting from the applied
			operation.
		"""
		raise NotImplementedError

	###########################################################################
	def preprocess_loss(self, model, loss):
		""" Validate / pre-process the loss functions before instantiating
			them.

			# Arguments

			model: The Kur model. It must be compiled.
			loss: Loss instance, list/tuple of Loss instances, or a dictionary
				of model layer names mapped to Loss instances.
		"""
		if not loss:
			num_outputs = len(model.outputs)
			logger.error('You are trying to construct a training/validation'
				'/testing model, but you haven\'t specified any loss '
				'functions. Your model has %d outputs: %s. You need to '
				'specify %d loss functions, one for each output.',
				num_outputs, ', '.join(model.outputs), num_outputs)
			raise ValueError('No loss functions were specified, but are '
				'required for training, testing, and validation.')

		if isinstance(loss, Loss):
			loss = [loss]

		output_only = set(model.outputs) - set(loss)
		if output_only:
			logger.debug('These layers will be output-only layers, without '
				'loss functions attached: %s', ', '.join(output_only))

		if isinstance(loss, (list, tuple)):
			if all(isinstance(x, Loss) for x in loss):
				if len(loss) != len(model.outputs):
					raise ValueError('Model has {} outputs, but only {} loss '
						'functions were specified.'
						.format(len(model.outputs), len(loss)))
				loss_with_names = list(zip(model.outputs, loss))
			elif all(isinstance(x, (dict, OrderedDict)) for x in loss):
				loss_with_names = []
				for x in loss:
					if not isinstance(x, (dict, OrderedDict)):
						raise ValueError('Expected each individual loss entry '
							'to be a dictionary. Received: {}'.format(x))
					if not 'target' in x:
						raise ValueError('Missing required key in loss '
							'function: "target".')
					loss_with_names.append((x['target'], x))
			else:
				raise ValueError('Expected a single loss function, a list of '
					'loss functions, a dictionary of loss functions, or a '
					'list of dictionaries. Instead we received: {}'
					.format(loss))

		elif isinstance(loss, (dict, OrderedDict)):
			loss_with_names = list(loss.items())

		else:
			raise ValueError('Loss functions given to "compile" should be '
				'a list/tuple, a dictionary, or a single Loss instance. '
				'Instead we received this: {} (type={})'
				.format(loss, type(loss)))

		return loss_with_names

	###########################################################################
	def compile(self, model, loss=None, optimizer=None, blocking=True,
		assemble_only=False):
		""" Creates an implementation-specific representation of the
			instantiated model.

			# Arguments

			model: Model instance. The built model to create a representation
				for.
			loss: Loss instance, list, dict, or None. The loss function
				(objective function) to use for training. If None, no loss
				functions are used. If a list, it should be a list of Loss
				instances, one for each model output; if the list is length 1,
				then the same loss function is used for each model output. If
				this is a dict, it should map from model output names to Loss
				instances to apply to that output.
			optimizer: Optimizer instance or None. The loss function to use for
				training. If None, no optimizer is used.
			blocking: bool (default: True). If True, requests that the
				implementation not return until all code compiling is complete.
				If False, the implementation may spawn background threads or
				processes, and these may still be running when
				train/test/evaluate are called (leading to blocking behavior at
				that point).
			assemble_only: bool (default: False). If True, does not attempt to
				create any compiled functions/models at all. Simply assembles
				the model and prepares the data sources. If this is True, then
				blocking is ignored.

			# Return value

			An implementation-specific object ("model") that can be used for
			training or evaluation. However, if `assemble_only` is True, then
			this returns a limited object which cannot be actually used for
			anything.

			# Notes

			Implementations may need to modify the Model. If this is the case,
			ExtensionState should be used to protect the model from extensions
			that are left stuck on after compiling.
		"""
		raise NotImplementedError

	###########################################################################
	def train(self, model, data):
		""" Fits the given model on a batch of data.

			# Arguments

			model: Model instance. The model to train with.
			data: dict or list. The data to train a batch on.
			compiled: object. The data package returned by `compile`. Its type
				and content are backend-specific.

			# Return value

			A dictionary of loss values, whose keys are the names of the
			respective outputs. If per-output loss is not available, then a
			single, global loss value can be returned instead.
		"""
		raise NotImplementedError

	###########################################################################
	def test(self, model, data):
		""" Calculates the model loss on a batch of data.

			# Arguments

			model: Model instance. The model to evaluate with.
			data: dict or list. The batch of data to test on.
			compiled: object. The data package returned by `compile`. Its type
				and content are backend-specific.

			# Return value

			A dictionary of loss values, whose keys are the names of the
			respective outputs. If per-output loss is not available, then a
			single, global loss value can be returned instead.
		"""
		raise NotImplementedError

	###########################################################################
	def evaluate(self, model, data, post_processor=None):
		""" Evaluates the model on a batch ofdata.
		"""
		raise NotImplementedError

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

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

from kur.utils import get_subclasses

###############################################################################
class EvaluationHook:					# pylint: disable=too-few-public-methods
	""" Base class for all evaluation hooks.

		Evaluation hooks are post-evaluation callbacks that can take a crack at
		modifying or reviewing the evaluation results.
	"""

	###########################################################################
	@staticmethod
	def from_specification(spec):
		""" Creates a new evaluation hook from a specification.

			# Arguments

			spec: str or dict. The specification object that was given.

			# Return value

			New EvaluationHook instance.
		"""
		if isinstance(spec, str):
			name = spec
			target = EvaluationHook.get_hook_by_name(spec)
			params = {}
			meta = {}
		elif isinstance(spec, dict):
			candidates = {}
			for key in spec:
				try:
					cls = EvaluationHook.get_hook_by_name(key)
					candidates[key] = cls
				except ValueError:
					pass

			if not candidates:
				raise ValueError('No valid evaluation hook found in '
					'Kurfile: {}'.format(spec))
			elif len(candidates) > 1:
				raise ValueError('Too many possible evaluation hooks found in '
					'Kurfile: {}'.format(spec))
			else:
				name, target = candidates.popitem()
				meta = dict(spec)
				meta.pop(name)
				params = spec[name] or {}

		else:
			raise ValueError('Expected the evaluation hooks to be a string or '
				'dictionary. We got this instead: {}'.format(spec))

		# At this point:
		# - name: the name of the evaluation hook, as specified in the spec.
		# - target: the class of the evaluation hook
		# - meta: dictionary of "other" keys in the spec that sit alongside the
		#   evaluation hook specification.
		# - params: parameters to be passed to the evaluation hook.

		return target(**params)

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the evaluation hook.

			# Return value

			A lower-case string unique to this evaluation hook.
		"""
		return cls.__name__.lower()

	###########################################################################
	@staticmethod
	def get_all_hooks():
		""" Returns an iterator to the names of all evaluation hooks.
		"""
		for cls in get_subclasses(EvaluationHook):
			yield cls

	###########################################################################
	@staticmethod
	def get_hook_by_name(name):
		""" Finds a evaluation hook class with the given name.
		"""
		name = name.lower()
		for cls in EvaluationHook.get_all_hooks():
			if cls.get_name() == name:
				return cls
		raise ValueError('No such evaluation hook with name "{}"'.format(name))

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new evaluation hook.
		"""
		if args or kwargs:
			raise ValueError('One or more unexpected or unsupported arguments '
				'to the evaluation hook: {}'.format(
					', '.join(args + list(kwargs.keys()))
			))

	###########################################################################
	def apply(self, current, original, model=None):
		""" Applies the hook to the data.

			# Arguments

			current: tuple. A 2-tuple, `(prediction, truth)`. `prediction` is
				a dictionary whose keys are the names of the output layers
				in the model, and whose values are a list of output tensors.
				If `truth` is not None, then it is a dictionary with the same
				keys as `prediction` which contains ground truth data.
			original: tuple. Same format as `current`. See Notes for details.
				This should not be modified.

			# Return value

			If the hook wants to modify the truth value that the next hook in
			the processing stack sees, then it should return a tuple of
			`(new_prediction, new_truth)` whose values are dictionaries in the
			same form as `data`. If the hook does not modify the truth data,
			then it can also simply return `new_prediction`.

			# Notes

			Hooks can be stacked. The value of `current` is the output of the
			most recent hook in the stack. `original` is the output of the
			model before being passed to the hooks. For the first hook, it is
			true that `output is current`.
		"""
		raise NotImplementedError

###############################################################################
class TrainingHook:					# pylint: disable=too-few-public-methods
	""" Base class for all training hooks.

		Training hooks are between-epoch callbacks that can be used to notify
		external systems about progress.
	"""

	TRAINING_START = object()
	TRAINING_END = object()
	VALIDATION_END = object()
	EPOCH_START = object()
	EPOCH_END = object()

	###########################################################################
	@staticmethod
	def from_specification(spec):
		""" Creates a new training hook from a specification.

			# Arguments

			spec: str or dict. The specification object that was given.

			# Return value

			New TrainingHook instance.
		"""
		if isinstance(spec, str):
			name = spec
			target = TrainingHook.get_hook_by_name(spec)
			params = {}
			meta = {}
		elif isinstance(spec, dict):
			candidates = {}
			for key in spec:
				try:
					cls = TrainingHook.get_hook_by_name(key)
					candidates[key] = cls
				except ValueError:
					pass

			if not candidates:
				raise ValueError('No valid training hook found in '
					'Kurfile: {}'.format(spec))
			elif len(candidates) > 1:
				raise ValueError('Too many possible training hooks found in '
					'Kurfile: {}'.format(spec))
			else:
				name, target = candidates.popitem()
				meta = dict(spec)
				meta.pop(name)
				params = spec[name] or {}

		else:
			raise ValueError('Expected the training hooks to be a string or '
				'dictionary. We got this instead: {}'.format(spec))

		if isinstance(params, dict):
			return target(**params)
		else:
			return target(params)

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the training hook.

			# Return value

			A lower-case string unique to this training hook.
		"""
		return cls.__name__.lower()

	###########################################################################
	@staticmethod
	def get_all_hooks():
		""" Returns an iterator to the names of all training hooks.
		"""
		for cls in get_subclasses(TrainingHook):
			yield cls

	###########################################################################
	@staticmethod
	def get_hook_by_name(name):
		""" Finds a training hook class with the given name.
		"""
		name = name.lower()
		for cls in TrainingHook.get_all_hooks():
			if cls.get_name() == name:
				return cls
		raise ValueError('No such training hook with name "{}"'.format(name))

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new training hook.
		"""
		if args or kwargs:
			raise ValueError('One or more unexpected or unsupported arguments '
				'to the training hook: {}'.format(
					', '.join(args + list(kwargs.keys()))
			))

	###########################################################################
	def notify(self, status, log=None, info=None):
		""" Notifies the hook that the epoch has ended.

			# Arguments

			status: object. One of the class objects (e.g., `TRAINING_START`)
				that indicates what is going on right now.
			log: Logger instance or None. The logger that has been tracking
				training and validation statistics.
			info: dict or None. Additional information to pass to the hook.
		"""
		raise NotImplementedError

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

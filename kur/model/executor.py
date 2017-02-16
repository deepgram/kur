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
import logging
import shutil
import math
import time
import traceback
import numpy
import tqdm
from ..utils import get_any_value, CriticalSection, parallelize
from ..loggers import PersistentLogger
from .hooks import TrainingHook

logger = logging.getLogger(__name__)

###############################################################################
class RetryException(Exception):
	""" Exception class for retrying an operation on a new batch of data.
	"""
	pass

###############################################################################
class Executor:
	""" Class for using models.
	"""

	MAX_RETRIES = 3

	###########################################################################
	def __init__(self, model, loss=None, optimizer=None):
		""" Creates a new executor.

			# Arguments

			model: Model instance. The model to train.
			loss: Loss instance. The loss function to use in training/testing.
			optimizer: Optimizer instance. The optimizer to use in training.
		"""
		self.model = model
		self.loss = loss
		self.optimizer = optimizer

	###########################################################################
	def compile(self, target=None, recompile=False, with_provider=None,
		**kwargs):
		""" Compiles a model.

			This generates a backend-specific representation of the model,
			suitable for training.

			# Arguments

			recompile: bool (default: False). If the model has already been
				compiled, it is not compiled again unless this flag is True.
			with_provider: Provider instance or None (default: None). If you
				want to merge the model's auxiliary data sources into your
				provider, you can specify the Provider instance here.

			# Return value

			None
		"""

		if target is None:
			if self.loss is None and self.optimizer is None:
				target = 'evaluate'
			elif self.optimizer is None:
				target = 'test'
			else:
				target = 'train'

		if not recompile:
			if self.model.compiled is not None \
				and target in self.model.compiled:
				return

		if not self.model.is_built():
			logger.warning('This model has never been built before. We are '
				'going to try to build it now. But the model should always be '
				'built with Model.build() before trying to compile it, just '
				'to ensure that everything has been parsed as you expect.')
			if with_provider is not None:
				self.model.register_provider(with_provider)
			self.model.build()

		logger.debug('Recompiling the model.')
		self.model.backend.compile(
			model=self.model,
			loss=self.loss if target != 'evaluate' else None,
			optimizer=None if target != 'train' else self.optimizer,
			blocking=True,
			**kwargs
		)

		if with_provider is not None:
			self.model.supplement_provider(with_provider)

	###########################################################################
	def test(self, provider, validating=False, hooks=None, step=False):
		""" Tests/validates the model on some data.

			# Arguments

			provider: Provider instance. The data provider which serves the
				data to be evaluated on.
			validating: bool (default: False). If False, the console output
				refers to this process as "testing"; otherwise, it is referred
				to as "validating."

			# Return value

			The average loss across the validation set.
		"""

		self.compile('test', with_provider=provider)

		if validating:
			desc = ('Validating', 'Validation')
		else:
			desc = ('Testing', 'Test')

		# Create progress bar
		test_loss = None
		n_entries = 0
		first_batch = None
		test_func = self.retry(self.model.backend.test)
		with tqdm.tqdm(
					total=len(provider),
					unit='samples',
					desc='{}, loss=N/A'.format(desc[0])
				) as pbar:

			# Present each batch to the network.
			for num_batches, batch in parallelize(enumerate(provider)):
				if step:
					self.do_step('Test', num_batches, batch)

				try:
					prediction, batch_loss = test_func(
						model=self.model,
						data=batch
					)
				except RetryException:
					continue

				if step and logger.isEnabledFor(logging.DEBUG):
					print(prediction)

				if first_batch is None:
					first_batch = (prediction, batch)

				batch_size = len(get_any_value(batch))

				#batch_loss = loss if isinstance(loss, float) \
				#	else sum(loss.values())

				new_entries = n_entries + batch_size

				if test_loss is None:
					test_loss = batch_loss
				else:
					test_loss = {
						k : v * (n_entries / new_entries) + \
							batch_loss[k] * (batch_size / new_entries)
						for k, v in test_loss.items()
					}
				#avg_loss = avg_loss * (n_entries / new_entries) + \
				#	batch_loss * (batch_size / new_entries)
				n_entries = new_entries

				# Update the progress bar
				pbar.set_description('{}, loss={:.3f}'.format(
					desc[0],
					sum(test_loss.values())
				))
				pbar.update(batch_size)

		if not n_entries:
			logger.warning('No data provided to validation/testing system.')
			return None

		logger.info('%s loss: %.3f', desc[1], sum(test_loss.values()))

		if hooks and first_batch is not None:
			prediction, batch = first_batch
			prev = first_batch
			for hook in hooks:
				new_prev = hook.apply(prev, first_batch, self.model)
				prev = (new_prev, prev[1]) \
					if not isinstance(new_prev, tuple) else new_prev

		return test_loss

	###########################################################################
	def train(self, *args, last_weights=None, log=None, training_hooks=None,
		**kwargs):
		""" Trains the model on some data.

			This is the public entry point for training. It wraps the business
			logic so that it can handle error conditions.
		"""

		reason = 'unknown'
		try:
			result = self.wrapped_train(
				*args,
				log=log,
				training_hooks=training_hooks,
				**kwargs
			)
		except (KeyboardInterrupt, Exception) as exc:
			logger.exception('Exception raised during training.')
			reason = traceback.format_exception_only(type(exc), exc)[0].strip()
			raise
		else:
			reason = 'success'
			return result
		finally:
			if last_weights is not None:
				logger.info('Saving most recent weights: %s', last_weights)
				with CriticalSection():
					self.model.save(last_weights)
			if log is not None:
				log.flush()

			if training_hooks:
				for hook in training_hooks:
					hook.notify(
						TrainingHook.TRAINING_END,
						log=log,
						info={'Reason' : reason}
					)

	###########################################################################
	def wrapped_train(self, provider, *, validation=None, epochs=None,
		log=None, best_train=None, best_valid=None, training_hooks=None,
		validation_hooks=None, checkpoint=None, step=False):
		""" Trains the model on some data.

			# Arguments

			provider: Provider instance. The data provider which serves the
				data to be trained on.
			validation: Provider instance or None (default: None). The data
				provider which serves validation data.
			epochs: int or None (default: None). The number of epochs to train
				for, or None to train forever.
			log: Log instance or None (default: None). The logger to save
				training statistics with.

			# Return value

			None
		"""

		#######################################################################
		# Process checkpoint requirements
		if isinstance(checkpoint, dict):
			if 'path' not in checkpoint:
				checkpoint['path'] = 'checkpoint'

			found = False
			for k in ('epochs', 'batches', 'samples'):
				if k in checkpoint:
					if not isinstance(checkpoint[k], int):
						raise ValueError('Expected "{}" key in "checkpoint" '
							'to be an integer. Received: {}'.format(k,
							checkpoint[k]))
					found = True

			if not found:
				checkpoint['epochs'] = 1

		elif isinstance(checkpoint, str):
			checkpoint = {
				'path' : checkpoint,
				'epochs' : 1
			}
		elif checkpoint is not None:
			raise ValueError('Unknown format for "checkpoint". Expected a '
				'single file or a dictionary. Instead we received: {}'
				.format(checkpoint))

		#######################################################################
		# Parse logs
		if log is None:
			logger.info('No log specified, so no historical loss information '
				'is available.')
			best_train_loss = best_valid_loss = None
		elif not isinstance(log, PersistentLogger):
			logger.info('Log type is non-persistent, so no historical loss '
				'information is available.')
			best_train_loss = best_valid_loss = None
		else:
			best_train_loss = log.get_best_training_loss()
			if best_train_loss is not None:
				logger.info('Best historical training loss: %.3f',
					best_train_loss)
			else:
				logger.info('No historical training loss available from logs.')

			best_valid_loss = log.get_best_validation_loss()
			if best_valid_loss is not None:
				logger.info('Best historical validation loss: %.3f',
					best_valid_loss)
			else:
				logger.info(
					'No historical validation loss available from logs.')

		#######################################################################
		# Parse desired number of epochs
		completed_epochs = log.get_number_of_epochs() if log else 0
		if not completed_epochs:
			logger.info('No previous epochs.')
		else:
			logger.info('Restarting from epoch %d.', completed_epochs+1)

		valid_modes = ('total', 'additional')
		default_mode = 'additional'
		mode = default_mode
		if isinstance(epochs, dict):
			mode = epochs.get('mode', default_mode)
			if mode not in valid_modes:
				raise ValueError('If "mode" in "epochs" must be one of: {}. '
					'Instead, we received: {}.'.format(', '.join(valid_modes),
					mode))
			if mode == 'total' and log is None:
				logger.warning('The epoch specification has "mode" set to '
					'"%s". This mode requires a log to be used correctly. Kur '
					'will proceed as if "mode" were "%s".', mode, default_mode)
				mode = default_mode
			epochs = epochs.get('number')
			if epochs in ('inf', 'all', 'infinite', 'infinity'):
				epochs = None
		elif not isinstance(epochs, (int, type(None))):
			raise ValueError('Expected "epochs" to be a dictionary or '
				'integer. Instead, we received: {}.'.format(epochs))
		logger.debug('Epoch handling mode: %s', mode)

		if epochs is not None:
			if mode == 'additional':
				epochs += completed_epochs

		#######################################################################
		# Local variables

		# The name of the most recently saved weight file. If the weights
		# change, this should be reset to None. Otherwise, saving weights can
		# be as simple as copying the previously saved file.
		saved_recent = None

		session = {
			'epochs' : 0,
			'batches' : 0,
			'samples' : 0,
			'minutes' : time.perf_counter() / 60
		}
		last_checkpoint = session.copy()

		epoch = completed_epochs - 1
		train_func = self.retry(self.model.backend.train)

		#######################################################################
		def run_validation(num_batches=None):
			""" Executes a validation run.
			"""
			if validation is None:
				return None

			nonlocal best_valid_loss

			# Continue with a validation run.
			try:
				if num_batches is not None and \
						hasattr(validation, 'num_batches'):
					previous_num_batches = validation.num_batches
					validation.num_batches = num_batches

				validation_loss = self.test(
					provider=validation,
					validating=True,
					hooks=validation_hooks
				)
			finally:
				if num_batches is not None and \
						hasattr(validation, 'num_batches'):
					validation.num_batches = previous_num_batches

			if validation_loss is None:
				return None

			cur_validation_loss = sum(validation_loss.values())
			if best_valid is not None:
				if best_valid_loss is None or \
						cur_validation_loss < best_valid_loss:
					logger.info(
						'Saving best historical validation weights: %s',
						best_valid
					)
					best_valid_loss = cur_validation_loss
					save_or_copy_weights(best_valid)

			if log is not None:
				log.log_validation(validation_loss, 'loss')

			return validation_loss

		#######################################################################
		def save_or_copy_weights(target):
			""" Saves the current model weights.
			"""
			nonlocal saved_recent

			if saved_recent is None:
				logger.debug('Saving weights to: %s', target)
				with CriticalSection():
					self.model.save(target)
				saved_recent = target
			elif not os.path.exists(saved_recent):
				logger.warning('Recently saved weight file seems to have '
					'vanished: %s', saved_recent)
				saved_recent = None
				save_or_copy_weights(target)
			elif os.path.exists(target) and \
					os.path.samefile(target, saved_recent):
				logger.debug('Recent weight file seems the same as the '
					'soon-to-be-saved file. Skipping: %s', target)
			else:
				logger.debug('Copying weights from: %s', saved_recent)
				with CriticalSection():
					shutil.rmtree(target, ignore_errors=True)
					shutil.copytree(saved_recent, target)

		#######################################################################
		def run_posttrain(n_entries, train_loss):
			""" Calculates training loss and saves if necessary.

				Read-only non-locals:
					n_entries, train_loss, best_train, log
				Read-write non-locals:
					best_train_loss
			"""
			nonlocal best_train_loss
			if not n_entries:
				logger.warning('No data provided to training loop.')
				return None

			cur_train_loss = sum(train_loss.values())
			logger.info('Training loss: %.3f', cur_train_loss)

			if best_train is not None:
				if best_train_loss is None or \
					cur_train_loss < best_train_loss:

					logger.info('Saving best historical training weights: '
						'%s', best_train)
					best_train_loss = cur_train_loss
					save_or_copy_weights(best_train)

			if log is not None:
				log.log_training(train_loss, 'loss')

			return cur_train_loss

		#######################################################################
		def run_training_hooks(cur_train_loss, validation_loss, status):
			""" Executes the training hooks, if necessary.

				Read-only non-locals:
					training_hooks, epoch, epochs, validation_loss
			"""
			if not training_hooks:
				return
			info = {
				'epoch' : epoch+1,
				'total_epochs' : epochs,
				'Training loss' : cur_train_loss
			}
			if validation is not None:
				info['Validation loss'] = validation_loss
			for hook in training_hooks:
				hook.notify(
					status,
					log=log,
					info=info
				)

		#######################################################################
		def run_checkpoint(*triggers, allow_validation=True):
			""" Runs the checkpoint triggers, if necessary.
			"""
			nonlocal last_checkpoint

			if checkpoint is None:
				return

			for k in triggers:
				if k not in checkpoint:
					continue
				if session[k] - last_checkpoint[k] >= checkpoint[k]:
					# We need a checkpoint

					# Save the file if necessary.
					if checkpoint['path']:
						logger.info('Making checkpoint backup: %s',
							checkpoint['path'])
						save_or_copy_weights(checkpoint['path'])

					# Validate if necessary.
					if checkpoint.get('validation', False) \
							and allow_validation:
						if isinstance(checkpoint['validation'], bool):
							num_batches = None
						else:
							num_batches = checkpoint['validation']
						val_loss = run_validation(num_batches)
						run_training_hooks(None, val_loss,
							TrainingHook.VALIDATION_END)

					last_checkpoint = session.copy()
					break

		#######################################################################
		# Prepare to train

		self.compile('train', with_provider=provider)
		provider.source_shapes()

		if training_hooks:
			for hook in training_hooks:
				hook.notify(
					TrainingHook.TRAINING_START,
					log=log
				)

		#######################################################################
		# Main training loop.
		while True:
			epoch += 1
			if epochs is not None and epoch >= epochs:
				print('Completed {} epochs.'.format(epochs))
				break

			print()

			###################################################################
			# START: Train one epoch

			# Create progress bar
			train_loss = None
			n_entries = 0
			with tqdm.tqdm(
						total=len(provider),
						unit='samples',
						desc='Epoch {}/{}, loss=N/A'
							.format(epoch+1, epochs or 'inf')
					) as pbar:

				# Present each batch to the network.
				for num_batches, batch in parallelize(enumerate(provider)):

					# The loss averaged over this batch.
					logger.debug('Training on batch...')
					if step:
						self.do_step(
							'Train, Epoch {}'.format(session['epochs']+1),
							num_batches, batch)

					try:
						prediction, batch_loss = train_func(
							model=self.model, data=batch)
					except RetryException:
						continue

					if step and logger.isEnabledFor(logging.DEBUG):
						print(prediction)

					# We just modified the weights. Invalidate the name of the
					# last weight file.
					saved_recent = None

					logger.debug('Finished training on batch.')

					# How many entries we just processed.
					batch_size = len(get_any_value(batch))

					if log is not None:
						log.log_batch(batch_size, batch_loss, 'loss')

					# Update our session statistics.
					session['batches'] += 1
					session['samples'] += batch_size
					session['minutes'] = time.perf_counter() / 60

					# Checkpoint if necessary
					run_checkpoint('samples', 'batches', 'minutes',
						allow_validation=True)

					# How many entries we've processed this epoch.
					new_entries = n_entries + batch_size

					# Average the per-batch loss across training.
					# This will give us our average "training loss".
					if train_loss is None:
						train_loss = batch_loss
					else:
						train_loss = {
							k : v * (n_entries / new_entries) + \
								batch_loss[k] * (batch_size / new_entries)
							for k, v in train_loss.items()
						}

					n_entries = new_entries

					# Update the progress bar with the current loss.
					# Note that `batch_loss` is, in some sense, just the
					# instantaneous training loss. `train_loss` is the average
					# loss across the entire training set so far.
					pbar.set_description('Epoch {}/{}, loss={:.3f}'.format(
						epoch+1, epochs or 'inf', sum(train_loss.values())
					))
					pbar.update(batch_size)

					for k, v in batch_loss.items():
						if math.isnan(v):
							logger.error('Received NaN loss value for '
								'model output "%s". Make sure that your '
								'inputs are all normalized and that the '
								'learning rate is not too high. Sometimes '
								'different algorithms/implementations '
								'work better than others, so you can try '
								'switching optimizers or backend.', k)
							raise ValueError('Model loss is NaN.')

			# END: Train one epoch
			###################################################################

			# Update our session statistics.
			session['epochs'] += 1

			# Checkpoint if necessary
			run_checkpoint('epochs', allow_validation=False)

			# Check to see what our current training loss is.
			cur_train_loss = run_posttrain(n_entries, train_loss)

			# Validate
			validation_loss = run_validation()

			# Execute training hooks.
			run_training_hooks(
				cur_train_loss,
				validation_loss,
				status=TrainingHook.EPOCH_END
			)

	###########################################################################
	def evaluate(self, provider, callback=None, step=False):
		""" Evaluates the model on some data.

			# Arguments

			provider: Provider instance. The data provider which serves the
				data to be evaluated.
			callback: function or None. If not None, the callback is called
				after each evaluation batch and is passed two parameters:
				`predicted` and `truth`, where `predicted` is the model output
				and `truth` is the ground truth data (if provided by
				`provider`; otherwise, `truth` is set to `None`).

			# Return value

			If `callback` is None, then this returns a tuple `(predicted,
			truth)`, where `predicted` is a dictionary whose keys are the names
			of the output nodes of the model, and whose respective values are
			arrays of predictions (one row per input sample). If the provider
			provides ground truth information, then `truth` has a similar
			structure to `predicted`; if ground truth information is not
			available, then `truth` is None.

			Otherwise, if `callback` is not None, this returns None.
		"""

		self.compile('evaluate', with_provider=provider)

		result = None
		truth = None
		has_truth = None
		total = len(provider)
		n_entries = 0

		eval_func = self.retry(self.model.backend.evaluate)

		with tqdm.tqdm(
					total=total,
					unit='samples',
					desc='Evaluating'
				) as pbar:

			for num_batches, batch in parallelize(enumerate(provider)):

				if step:
					self.do_step('Evaluate', num_batches, batch)

				try:
					evaluated, _ = eval_func(model=self.model, data=batch)
				except RetryException:
					continue

				if step and logger.isEnabledFor(logging.DEBUG):
					print(evaluated)

				batch_size = len(get_any_value(batch))

				if has_truth is None:
					has_truth = all(k in batch for k in self.model.outputs)

				if callback is None:
					# There is no callback. We need to hang on to everything.
					if total is None:
						# We don't know how many entries there will be.
						if result is None:
							# This is our first batch.
							result = {k : [] for k in self.model.outputs}
						for k, v in evaluated.items():
							result[k].extend(v)

						if has_truth:
							if truth is None:
								truth = {k : [] for k in self.model.outputs}
							for k in truth:
								truth[k].extend(batch[k])
					else:
						# We know how many entries there will be.
						if result is None:
							# This is our first batch.
							result = {k : [None]*total for k in evaluated}
						for k, v in evaluated.items():
							result[k][n_entries:(n_entries+batch_size)] = v[:]

						if has_truth:
							if truth is None:
								truth = {k : [None]*total for k in evaluated}
							for k in truth:
								truth[k][n_entries:(n_entries+batch_size)] = \
									batch[k][:]
				else:
					callback(evaluated, truth)

				n_entries += batch_size
				pbar.update(batch_size)

		if callback is not None:
			return

		if total is None:
			for k, v in result.items():
				result[k] = numpy.concatenate(v)
			for k, v in truth.items():
				truth[k] = numpy.concatenate(v)

		return result, truth

	###########################################################################
	def do_step(self, what, num_batches, batch):
		""" Wait for user input before running a single batch of data.
		"""
		print('{}, Batch {}:'.format(what, num_batches+1))
		if logger.isEnabledFor(logging.DEBUG):
			for k, v in batch.items():
				print('{} {}: {}'.format(
					k,
					v.shape if hasattr(v, 'shape') else \
						'(list, {} entries)'.format(len(v)),
					v
				))
		input('Press ENTER to continue...')

	###########################################################################
	def retry(self, func):
		""" Creates a wrapper that implements some retry semantics.
		"""

		def try_func(*args, **kwargs):
			""" Wraps a function with some retry logic.
			"""
			try:
				result = func(*args, **kwargs)

			# Catch Exception so that we don't catch KeyboardInterrupt.
			except Exception:
				try_func.counter += 1
				if try_func.counter > Executor.MAX_RETRIES:
					logger.exception(
						'Failed to execute on batch. No more retries.')
					raise
				logger.exception('Failed to execute on batch. Tolerating up '
					'to %d more consecutive failures.',
					Executor.MAX_RETRIES - try_func.counter)
				raise RetryException
			else:
				try_func.counter = 0
				return result
		try_func.counter = 0

		return try_func

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

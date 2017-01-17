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

import logging
import shutil
import tempfile
import math
import tqdm
from ..utils import get_any_value, CriticalSection, parallelize

logger = logging.getLogger(__name__)

###############################################################################
class Executor:
	""" Class for using models.
	"""

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
	def compile(self, target, recompile=False, with_provider=None):
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
			blocking=True
		)

		if with_provider is not None:
			self.model.supplement_provider(with_provider)

	###########################################################################
	def test(self, provider, validating=False, hooks=None):
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
		with tqdm.tqdm(
					total=len(provider),
					unit='samples',
					desc='{}, loss=N/A'.format(desc[0])
				) as pbar:

			# Present each batch to the network.
			for batch in parallelize(provider):
				prediction, batch_loss = self.model.backend.test(
					model=self.model,
					data=batch
				)

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
			for hook in hooks:
				prediction = hook.apply(prediction, batch, self.model)

		return test_loss

	###########################################################################
	def train(self, *args, last_weights=None, log=None, **kwargs):
		""" Trains the model on some data.

			This is the public entry point for training. It wraps the business
			logic so that it can handle error conditions.
		"""

		try:
			result = self.wrapped_train(
				*args,
				log=log,
				**kwargs
			)
		except:
			logger.exception('Exception raised during training.')
			raise
		else:
			return result
		finally:
			if last_weights is not None:
				logger.info('Saving most recent weights: %s', last_weights)
				with CriticalSection():
					self.model.save(last_weights)
			if log is not None:
				log.flush()

	###########################################################################
	def wrapped_train(self, provider, *, validation=None, epochs=None,
		log=None, best_train=None, best_valid=None, validation_hooks=None):
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

		self.compile('train', with_provider=provider)
		provider.source_shapes()

		if log is None:
			logger.info('No log specified, so no historical loss information '
				'is available.')
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

		# The name of the most recently saved weight file. If the weights
		# change, this should be reset to None. Otherwise, saving weights can
		# be as simple as copying the previously saved file.
		saved_recent = None

		epoch = -1
		while True:
			epoch += 1
			if epochs is not None and epoch >= epochs:
				break

			# We are about to modify the weights. Invalidate the name of the
			# last weight file.
			saved_recent = None

			print()

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
				for batch in parallelize(provider):

					# The loss averaged over this batch.
					logger.debug('Training on batch...')
					_, batch_loss = self.model.backend.train(
						model=self.model,
						data=batch
					)
					logger.debug('Finished training on batch.')

					if log is not None:
						log.log_batch(batch_loss, 'loss')

					# How many entries we just processed.
					batch_size = len(get_any_value(batch))

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
								'model output "%s".', k)
							return

			if not n_entries:
				logger.warning('No data provided to training loop. Trying to '
					'move on to the next epoch.')
				continue

			cur_train_loss = sum(train_loss.values())
			logger.info('Training loss: %.3f', cur_train_loss)

			if best_train is not None:
				if best_train_loss is None or cur_train_loss < best_train_loss:
					logger.info('Saving best historical training weights: %s',
						best_train)
					best_train_loss = cur_train_loss
					with CriticalSection():
						self.model.save(best_train)
					saved_recent = best_train

			if log is not None:
				log.log_training(train_loss, 'loss')

			if validation is not None:
				# Continue with a validation run.
				validation_loss = self.test(
					provider=validation,
					validating=True,
					hooks=validation_hooks
				)
				if validation_loss is None:
					continue

				cur_validation_loss = sum(validation_loss.values())
				if best_valid is not None:
					if best_valid_loss is None or \
							cur_validation_loss < best_valid_loss:
						logger.info(
							'Saving best historical validation weights: %s',
							best_valid
						)
						best_valid_loss = cur_validation_loss
						if saved_recent is None:
							with CriticalSection():
								self.model.save(best_valid)
							saved_recent = best_valid
						else:
							logger.debug(
								'Copying weights from: %s',
								saved_recent
							)
							with CriticalSection():
								shutil.rmtree(best_valid, ignore_errors=True)
								shutil.copytree(saved_recent, best_valid)

				if log is not None:
					log.log_validation(validation_loss, 'loss')

	###########################################################################
	def evaluate(self, provider, callback=None):
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

		with tqdm.tqdm(
					total=total,
					unit='samples',
					desc='Evaluating'
				) as pbar:

			for batch in parallelize(provider):
				evaluated, _ = self.model.backend.evaluate(
					model=self.model,
					data=batch
				)
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

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

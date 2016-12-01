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
import tqdm
from ..utils import get_any_value, CriticalSection

logger = logging.getLogger(__name__)

################################################################################
class Trainer:
	""" Class for training models.
	"""

	############################################################################
	def __init__(self, model, loss, optimizer=None):
		""" Creates a new trainer.

			# Arguments

			model: Model instance. The model to train.
			loss: Loss instance. The loss function to use in training.
			optimizer: Optimizer instance. The optimizer to use in training.
		"""
		self.model = model
		self.loss = loss
		self.optimizer = optimizer

		self._compiled = None

	############################################################################
	def compile(self, recompile=False):
		""" Compiles a model.

			This generates a backend-specific representation of the model,
			suitable for training.

			# Arguments

			recompile: bool (default: False). If the model has already been
				compiled, it is not compiled again unless this flag is True.

			# Return value

			None
		"""

		if self._compiled is not None and not recompile:
			return

		if not self.model.is_built():
			logger.warning('This model has never been built before. We are '
				'going to try to build it now. But the model should always be '
				'built with Model.build() before trying to compile it, just to '
				'ensure that everything has been parsed as you expect.')
			self.model.build()

		logger.debug('Recompiling the model.')
		self._compiled = self.model.backend.compile(
			model=self.model,
			loss=self.loss,
			optimizer=self.optimizer
		)

	############################################################################
	def test(self, provider, validating=False):
		""" Tests/validates the model on some data.

			# Arguments

			provider: Provider instance. The data provider which serves the data
				to be evaluated on.
			validating: bool (default: False). If False, the console output
				refers to this process as "testing"; otherwise, it is referred
				to as "validating."

			# Return value

			The average loss across the validation set.
		"""

		self.compile()

		if validating:
			desc = ('Validating', 'Validation')
		else:
			desc = ('Testing', 'Test')

		# Create progress bar
		test_loss = None
		n_entries = 0
		with tqdm.tqdm(
					total=len(provider),
					unit='samples',
					desc='{}, loss=N/A'.format(desc[0])
				) as pbar:

			# Present each batch to the network.
			for batch in provider:
				batch_loss = self.model.backend.test(
					model=self.model,
					data=batch,
					compiled=self._compiled
				)

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

		return test_loss

	############################################################################
	def train(self, *args, last_weights=None, log=None, **kwargs):

		try:
			result = self.wrapped_train(*args, log=log, **kwargs)
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

	############################################################################
	def wrapped_train(self, provider, validation=None, epochs=None, log=None,
		best_train=None, best_valid=None):
		""" Trains the model on some data.

			# Arguments

			provider: Provider instance. The data provider which serves the data
				to be trained on.
			validation: Provider instance or None (default: None). The data
				provider which serves validation data.
			epochs: int or None (default: None). The number of epochs to traing
				for, or None to train forever.
			log: Log instance or None (default: None). The logger to save
				training statistics with.

			# Return value

			None
		"""

		self.compile()

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
						desc='Epoch {}/{}, loss=N/A'.format(epoch+1, epochs)
					) as pbar:

				# Present each batch to the network.
				for batch in provider:
					# The loss averaged over this batch.
					batch_loss = self.model.backend.train(
						model=self.model,
						data=batch,
						compiled=self._compiled
					)

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
						epoch+1, epochs, sum(train_loss.values())
					))
					pbar.update(batch_size)

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
					validating=True
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
								shutil.copy(saved_recent, best_valid)

				if log is not None:
					log.log_validation(validation_loss, 'loss')

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

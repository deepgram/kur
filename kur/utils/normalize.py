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

import logging
import warnings

import yaml
import numpy

logger = logging.getLogger(__name__)

###############################################################################
class Normalize:
	""" Class for normalizing data (centering, whitening, re-scaling, etc.)
	"""

	APPROXIMATION_CUTOFF = 10000

	###########################################################################
	def __init__(self, center=True, scale=True, rotate='zca'):
		""" Creates a new identity transform.
		"""
		if rotate is True:
			rotate = 'zca'
		if isinstance(rotate, str) and rotate not in ('zca', 'pca'):
			raise ValueError('Invalid whitening transformation: {}'
				.format(rotate))

		if rotate == 'zca' and not scale:
			warnings.warn('"zca" without scaling is an identity transform. '
				'Only "center" will have any effect.', UserWarning)

		self.center = center
		self.scale = scale
		self.rotate = rotate
		self.state = None
		self.transform = None

	###########################################################################
	def apply(self, data):
		""" Applies the normalization to a single sample.
		"""
		if self.transform is None:
			logger.warning('Normalization transform has not been loaded. '
				'Using an identity transform instead. Be sure to call '
				'`Normalize.learn()` before trying to use `apply()`!')
			return data
		return self.transform(data)

	###########################################################################
	def has_learned(self):
		""" Returns True if we have a normalization transform available.
		"""
		return self.state is not None

	###########################################################################
	def get_dimensionality(self):
		""" Returns the dimensionality of the normalization.
		"""
		return self.state['mean'].shape[0]

	###########################################################################
	def learn(self, data):
		""" Learns the normalization with the current data stream.

			# Arguments

			data: list of data vectors (or matrices)
		"""

		if len(data) > Normalize.APPROXIMATION_CUTOFF:
			logger.warning('Data set is large. Normalization will be '
				'approximate.')
			indices = numpy.random.permutation(
				len(data)
			)[:Normalize.APPROXIMATION_CUTOFF]
			data = data[indices]

		data = numpy.vstack(data)
		mean = data.mean(axis=0)

		stddev = data.std(axis=0, ddof=1)
		stddev = numpy.where(stddev, stddev, 1.)

		centered = data - mean

		# pylint: disable=invalid-name

		# Figure out if we need the full matrices
		M, N = centered.shape
		_, S, Vt = numpy.linalg.svd(centered, full_matrices=bool(M < N))

		s = numpy.array([1/s if s > S[0] * 1e-6 else 0. for s in S])
		s *= numpy.sqrt(data.shape[0] - 1)
		S = numpy.diag(s)

		# pylint: enable=invalid-name

		self.state = {
			'mean' : mean,
			'stddev' : stddev,
			'S' : S,
			'Vt' : Vt
		}

		self.transform = self._build_transform()

	###########################################################################
	def _build_transform(self):
		""" Creates the normalization transform function.
		"""
		if self.rotate:
			current_transform = self.state['Vt']

			# This is not mathematically necessary, but PCA involves rotation
			# around eigenvectors, which are only unique up to an overall sign.
			# So to make things repeatable
			if self.rotate == 'pca':
				reflect = numpy.array(
					[-1 if r[0] < 0 else 1 for r in current_transform]
				)
				current_transform = reflect * current_transform

			if self.scale:
				current_transform = self.state['S'].dot(current_transform)

			if self.rotate == 'zca':
				current_transform = (self.state['Vt'].T).dot(current_transform)

			if self.center:
				return lambda x: (x - self.state['mean']).dot(
					current_transform.T)
			else:
				return lambda x: ((x - self.state['mean']).dot(
					current_transform.T) + self.state['mean'])

		if self.scale:
			if self.center:
				return lambda x: \
					(x - self.state['mean']) / self.state['stddev']
			else:
				return lambda x: ( \
					(x - self.state['mean']) / self.state['stddev'] + \
					self.state['mean']
				)

		if self.center:
			return lambda x: x - self.state['mean']
		else:
			return lambda x: x

	###########################################################################
	def save(self, filename):
		""" Saves normalization statistics.
		"""
		logger.info('Saving normalization state to: %s', filename)
		if self.state is None:
			to_save = {}
		else:
			to_save = {
				key : value.tolist() if hasattr(value, 'tolist') else value
				for key, value in self.state.items()
			}

		with open(filename, 'w') as fh:
			fh.write(yaml.dump(to_save))

	###########################################################################
	def restore(self, filename):
		""" Loads previously saved normalization statistics.
		"""
		logger.info('Restoring normalization state from: %s', filename)
		with open(filename, 'r') as fh:
			to_load = yaml.load(fh.read())

		if not to_load:
			self.state = None
			self.transform = None
		else:
			self.state = {
				key : numpy.array(value) if isinstance(value, (tuple, list)) \
					else value
				for key, value in to_load.items()
			}
			self.transform = self._build_transform()

	###########################################################################
	def get_state(self):
		""" Getter for the state.
		"""
		return self.state

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

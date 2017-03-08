"""
Copyright 2017 Anthony Rousseau for Allo-media
Based on kuza55 keras-extras (https://github.com/kuza55/keras-extras)
and jonilaserson (https://github.com/fchollet/keras/issues/2436) code.

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

from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
from keras import backend as K

import tensorflow as tf


def make_parallel(model, gpu_count):
	""" Allows a model to run on multiple GPUs.
	"""
	def slice_batch(x, n_gpus, part):
		""" Divide the input batch into [n_gpus] slices,
			and obtain slice no. [part].
			i.e. if len(x) = 10, then slice_batch(x, 2, 1)
			will return x[5:].
		"""
		sh = K.shape(x)
		L = sh[0] // n_gpus # sh[0] = batch_size

		if part == n_gpus - 1:
			return x[part*L:]

		return x[part*L:(part+1)*L]

	all_outputs = []

	# Empty list for each output in the model
	for i in range(len(model.outputs)):
		all_outputs.append([])

	# Place a copy of the model on each GPU, 
	# each getting a slice of the batch
	for i in range(gpu_count):
		with tf.device('/gpu:%d' % i):
			with tf.name_scope('tower_%d' % i) as scope:
				inputs = []

				# Slice each input into a piece 
				# for processing on this GPU
				for x in model.inputs:
					input_shape = tuple(x.get_shape().as_list())[1:]
					slice_n = Lambda(slice_batch, 
						lambda shape: input_shape, 
						arguments={'n_gpus':gpu_count, 'part':i})(x)
					inputs.append(slice_n)

				outputs = model(inputs)

				if not isinstance(outputs, list):
					outputs = [outputs]

				# Save all the outputs for 
				# merging back together later
				for l in range(len(outputs)):
					all_outputs[l].append(outputs[l])

	# Merge outputs on CPU
	with tf.device('/cpu:0'):
		merged = []

		for outputs in all_outputs:
			merged.append(merge(outputs, mode='concat', concat_axis=0))

	return Model(input=model.inputs, output=merged)

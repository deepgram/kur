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

import gzip
import logging
import struct
import tempfile
import os
import hashlib

import requests
import kur
import numpy

logger = logging.getLogger(__name__)

################################################################################
def download_data(url, shasum):
	""" Downloads a URL to the system temporary directory.

		# Arguments

		url: str. The URL of the resource to download.
		shasum: str. The SHA256 hash of the resource content.

		# Return value

		The path of the file on the local system.

		# Notes

		This will only download the file if it doesn't already exist or if its
		checksum fails.
	"""

	_, filename = os.path.split(url)
	target = os.path.join(tempfile.gettempdir(), filename)

	if os.path.isfile(target):
		with open(target, 'rb') as fh:
			data = fh.read()
		sha = hashlib.sha256()
		sha.update(data)
		if sha.hexdigest() == shasum:
			print('Data present: {}'.format(target))
			return target

		print('Corrupt file present. Re-downloading: {}'.format(url))
	else:
		print('Downloading: {}'.format(url))

	data = requests.get(url)
	sha = hashlib.sha256()
	sha.update(data.content)
	if sha.hexdigest() != shasum:
		raise ValueError('Failed integrity check.')

	with open(target, 'wb') as fh:
		fh.write(data.content)

	return target

################################################################################
def load_labels(filename):
	""" Loads MNIST labels.

		# Arguments

		filename: str. The path to the file with the MNIST labels.

		# Return value

		A numpy array of one-hot labels.
	"""
	logger.debug('Loading MNIST labels: %s', filename)
	with gzip.open(filename) as fh:
		data = fh.read()

	magic, num_items = struct.unpack('>2I', data[0:8])
	if magic != 0x00000801:
		raise ValueError('Bad magic number.')

	logging.debug('Num items: %d', num_items)

	data = data[8:]
	if len(data) != num_items:
		logger.error('Expected %d items, got %d items', num_items, len(data))
		raise ValueError('Length mis-match.')

	data = numpy.fromstring(data, dtype='>u1').astype(numpy.int32)
	onehot = numpy.zeros((data.shape[0], 10))
	for i, row in enumerate(data):
		onehot[i][row] = 1
	return onehot

################################################################################
def load_images(filename):
	""" Loads MNST images.

		# Arguments

		filename: str. The path to the file containing the MNIST images.

		# Return value

		A numpy array of the images. The images will be mean-subtracted and
		normalized.
	"""
	logger.debug('Loading MNIST images: %s', filename)
	with gzip.open(filename) as fh:
		data = fh.read()

	magic, num_items, num_rows, num_cols = struct.unpack('>4I', data[0:16])
	if magic != 0x00000803:
		raise ValueError('Bad magic number.')

	logging.debug('Num items: %d', num_items)

	image_size = num_rows * num_cols
	data = data[16:]
	if len(data) != num_items * image_size:
		logger.error('Expected %d items, got %d items', num_items, len(data))
		raise ValueError('Length mis-match.')

	data = numpy.fromstring(data, dtype='>u1').astype(numpy.float32)
	data /= 256
	data -= data.mean()
	return data.reshape(num_items, num_rows, num_cols)

################################################################################
def get_training_data():
	""" Returns the training set.

		# Return value

		Dictionary with keys 'images' and 'labels' and values that are the
		MNIST normalized images and one-hot labels, respectively.
	"""
	images = load_images(download_data(
		'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
		'440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609'
	))

	labels = load_labels(download_data(
		'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
		'3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c'
	))

	return {'images' : images, 'labels' : labels}

################################################################################
def get_testing_data():
	""" Returns the testing set.

		# Return value

		Dictionary with keys 'images' and 'labels' and values that are the
		MNIST normalized images and one-hot labels, respectively.
	"""
	images = load_images(download_data(
		'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
		'8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6'
	))

	labels = load_labels(download_data(
		'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
		'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'
	))

	return {'images' : images, 'labels' : labels}

################################################################################
def main():									# pylint: disable=too-many-locals
	""" Training and evaluation example using the MNIST handwriting datset.

		# References

		- http://yann.lecun.com/exdb/mnist/
	"""

	# How much the of the data set to load.
	# 1 = all of it (100%)
	# 0.01 = 1% of it.
	# This is mostly useful on slower machines where it is more important to see
	# how the example works than to get good results.
	ratio = 0.01
	print('Using {:.0f}% of the data.'.format(ratio*100))

	# Where to save/load the model weights.
	weight_file = os.path.join(tempfile.gettempdir(), 'mnist.weights')

	# Load the specification.
	specification = kur.reader.Reader.read_file('mnist.yml')

	# Get the training data.
	data = get_training_data()

	# 50,000 training + 10,000 validation
	#     = 60,000 total images in the training set.
	training = {k : v[:int(50000*ratio)] for k, v in data.items()}
	validation = {k : v[-int(10000*ratio):] for k, v in data.items()}

	training_provider = kur.providers.BatchProvider(
		sources={
			'images' : kur.sources.VanillaSource(training['images']),
			'labels' : kur.sources.VanillaSource(training['labels'])
		},
		batch_size=32
	)

	validation_provider = kur.providers.BatchProvider(
		sources={
			'images' : kur.sources.VanillaSource(validation['images']),
			'labels' : kur.sources.VanillaSource(validation['labels'])
		},
		batch_size=32
	)

	# Create the Kur model.
	model = kur.model.Model(
		backend=kur.backend.Backend.get_any_supported_backend()(),
		containers=[kur.containers.Container.create_container_from_data(entry)
			for entry in specification['model']]
	)

	# Parse and build the model.
	engine = kur.engine.JinjaEngine()
	model.parse(engine)
	model.apply_provider_knowledge(training_provider)
	model.build()

	# Create the training object.
	trainer = kur.model.Trainer(
		model=model,
		loss=kur.loss.Loss.get_loss_by_name('categorical_crossentropy')(),
		optimizer=kur.optimizer.Optimizer.get_optimizer_by_name('adam')()
	)

	if os.path.isfile(weight_file):
		print('Loading weights from:', weight_file)
		model.restore(weight_file)

	# Train the model.
	trainer.train(
		provider=training_provider,
		validation=validation_provider,
		epochs=1,
		log=None
	)

	print('Saving weights to:', weight_file)
	model.save(weight_file)

	# Create the evaluation object.
	evaluator = kur.model.Evaluator(model=model)

	# Get the testing data.
	data = get_testing_data()

	data = {k : v[:int(10000*ratio)] for k, v in data.items()}

	# Evaluate the model.
	evaluated, _ = evaluator.evaluate(
		provider=kur.providers.BatchProvider(
			sources={
				'images' : kur.sources.VanillaSource(data['images'])
			},
			batch_size=32,
			randomize=False
		)
	)

	# Print out statistics.
	for k, v in evaluated.items():
		good = 0
		total = 0

		# Compare the predictions to the labels.
		for result in kur.providers.BatchProvider(
			sources={
				'labels' : kur.sources.VanillaSource(data['labels']),
				'evaluated' : kur.sources.VanillaSource(v)
			},
			randomize=False,
			batch_size=32
		):
			truth = result['labels'].argmax(axis=1)
			predict = result['evaluated'].argmax(axis=1)
			bad = numpy.count_nonzero(truth - predict)
			good += len(truth) - bad
			total += len(truth)

		print('{}: {}/{} correct'.format(k, good, total))

################################################################################
if __name__ == '__main__':
	main()

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF

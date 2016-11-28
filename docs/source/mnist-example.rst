Let's jump right in and see how awesome Kur is! The first example we'll look at
is Yann LeCun's `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset. This is a
dataset of 28x28 pixel images of individual handwritten digits between 0 and 9.
The goal of our model will be to perform image recognition, tagging the image
with the most likely digit it represents.

.. note::

	As with most command line examples, lines preceded by ``$`` are lines that
	you are supposed to type (followed by the ``ENTER`` key). Lines without an
	initial ``$`` are lines which are printed to the screen (you don't type
	them).

First, you need to :ref:`get_the_code`! If you install via ``pip``, you'll need
to checkout the ``examples`` directory from the repository; if you install via
``git``, then you alreay have the ``examples`` directory locally. So let's move
into the example directory:

.. code-block:: bash

	$ cd examples

Now let's train the MNIST model. This will download the data directly from the
web, and then start training for 10 epochs.

.. code-block:: bash
   :emphasize-lines: 1

	$ kur train mnist.yml
	Downloading: 100%|█████████████████████████████████| 9.91M/9.91M [03:44<00:00, 44.2Kbytes/s]
	Downloading: 100%|█████████████████████████████████| 28.9K/28.9K [00:00<00:00, 66.1Kbytes/s]
	Downloading: 100%|█████████████████████████████████| 1.65M/1.65M [00:31<00:00, 52.6Kbytes/s]
	Downloading: 100%|█████████████████████████████████| 4.54K/4.54K [00:00<00:00, 19.8Kbytes/s]

	Epoch 1/10, loss=1.750: 100%|███████████████████████| 320/320 [00:02<00:00, 154.81samples/s]
	Validating, loss=1.102: 100%|██████████████████| 10000/10000 [00:05<00:00, 1737.00samples/s]

	Epoch 2/10, loss=0.888: 100%|███████████████████████| 320/320 [00:01<00:00, 283.95samples/s]
	Validating, loss=0.666: 100%|██████████████████| 10000/10000 [00:08<00:00, 1209.40samples/s]

	Epoch 3/10, loss=0.551: 100%|███████████████████████| 320/320 [00:01<00:00, 269.09samples/s]
	Validating, loss=0.504: 100%|██████████████████| 10000/10000 [00:08<00:00, 1221.64samples/s]

	Epoch 4/10, loss=0.446: 100%|███████████████████████| 320/320 [00:01<00:00, 233.96samples/s]
	Validating, loss=0.438: 100%|██████████████████| 10000/10000 [00:08<00:00, 1174.40samples/s]

	Epoch 5/10, loss=0.544: 100%|███████████████████████| 320/320 [00:01<00:00, 269.47samples/s]
	Validating, loss=0.398: 100%|██████████████████| 10000/10000 [00:08<00:00, 1235.31samples/s]

	Epoch 6/10, loss=0.508: 100%|███████████████████████| 320/320 [00:01<00:00, 253.47samples/s]
	Validating, loss=0.409: 100%|██████████████████| 10000/10000 [00:08<00:00, 1243.92samples/s]

	Epoch 7/10, loss=0.464: 100%|███████████████████████| 320/320 [00:01<00:00, 263.46samples/s]
	Validating, loss=0.384: 100%|██████████████████| 10000/10000 [00:08<00:00, 1209.80samples/s]

	Epoch 8/10, loss=0.388: 100%|███████████████████████| 320/320 [00:01<00:00, 260.60samples/s]
	Validating, loss=0.375: 100%|██████████████████| 10000/10000 [00:08<00:00, 1230.72samples/s]

	Epoch 9/10, loss=0.485: 100%|███████████████████████| 320/320 [00:01<00:00, 278.96samples/s]
	Validating, loss=0.428: 100%|██████████████████| 10000/10000 [00:08<00:00, 1228.11samples/s]

	Epoch 10/10, loss=0.428: 100%|██████████████████████| 320/320 [00:01<00:00, 280.16samples/s]
	Validating, loss=0.360: 100%|██████████████████| 10000/10000 [00:08<00:00, 1225.70samples/s]

What just happened? Kur downloaded the MNIST dataset from LeCun's website, and
then trained a model for ten epochs. Awesome!

Now let's see how well our model actually performs:

.. code-block:: bash
   :emphasize-lines: 1

	$ kur evaluate mnist.yml
	Evaluating: 100%|██████████████████████████████| 10000/10000 [00:05<00:00, 1767.62samples/s]
	LABEL     CORRECT   TOTAL     ACCURACY  
	0         968       980        98.8%
	1         1097      1135       96.7%
	2         867       1032       84.0%
	3         931       1010       92.2%
	4         903       982        92.0%
	5         744       892        83.4%
	6         838       958        87.5%
	7         927       1028       90.2%
	8         860       974        88.3%
	9         825       1009       81.8%
	ALL       8960      10000      89.6%

Wow! Across the board, we already have about 90% accuracy for recognizing
handwritten digits. That's how awesome Kur is.

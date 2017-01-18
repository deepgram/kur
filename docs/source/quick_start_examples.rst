.. _quick_examples:

*********************
Examples: Quick Start
*********************

Let's look at some examples of how easy Kur makes state-of-the-art deep
learning.

.. note::

	In this section you can jump right in and train instantly without knowing what is REALLY going on. That's totally ok! Get hyped on Kur with the quick (and impressive) examples, then proceed to the :ref:`in_depth_examples` for a more thorough going through.

First, you need to :ref:`install_kur`! If you installed via
``pip``, you'll need to checkout the ``examples`` directory from the
repository, like this:

.. code-block:: bash

	git clone https://github.com/deepgram/kur
	cd kur/examples

If you installed via ``git``, then you alreay have the ``examples`` directory
locally, so just move into the example directory:

.. code-block:: bash

	$ cd examples


.. _quick_mnist_example:

MNIST: Handwriting recognition
==============================

Train the model on the MNIST dataset.

.. code-block:: bash

	kur train mnist.yml

Again, evaluation is just as simple:

.. code-block:: bash

	kur evaluate mnist.yml


.. _quick_cifar_10_example:

CIFAR-10: Image Classification
==============================

Great, we now have a simple, but powerful and general model. Let's train it. As
before, you'll need to ``cd examples`` first.

.. code-block:: bash

	kur train cifar.yml

Again, evaluation is just as simple:

.. code-block:: bash

	kur evaluate cifar.yml

.. _quick_speech_example:

DEEPGRAM10: Speech Recognition
==============================

This is the speech example with CNN layer and RNN stack.

.. code-block:: bash

	kur train speech.yml

Again, evaluation is just as simple:

.. code-block:: bash

	kur evaluate speech.yml
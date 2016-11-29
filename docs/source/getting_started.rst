*********
Using Kur
*********

Let's get started using Kur! There's two topics we need to cover: describing
models in a specification file, and actually running Kur.

.. note::

	If you haven't already, make sure you work through the :doc:`examples`
	first. Otherwise, this might feel a little overwhelming, and Kur shouldn't
	make you feel like that.

Specification Files
===================

Kur uses "specification" files to describe the model, hyperparameters, data
sets, training/evaluation options, and functional settings. This is a whirlwind
tour of how Kur interprets the specification files. For more details, see
:doc:`specification`. 

YAML Notes
----------

Since YAML is the default supported Kur format, here are a couple pointers.

- YAML documents need to start with three dashes: ``---``. Everything
  you add to the file should go below those dashes.
- Documents can be explicitly terminated by three periods: ``...``, but this is
  optional.
- YAML is a "whitespace matters" language. You should never use tabs in YAML.
- YAML files should use the ``.yaml`` or ``.yml`` extension.
- YAML comments start with the hash character: ``#``

For details of the YAML syntax, take a look at the `Ansible overview
<https://docs.ansible.com/ansible/YAMLSyntax.html>`_

Skeleton Outline of Simple Specification
----------------------------------------

This is a simplied template you can use for writing simple Kur models.

.. code-block:: yaml

	---

	# Other specification files to load (optional)
	include:

	# Global variables go here (optional)
	settings:

	# Your core model goes here (required)
	model:

	  # Input data
	  - input: INPUT

	  # ... other layers ...

	  # Last layer. Change "softmax" if it is appropriate.
	  - activation: softmax
	    name: OUTPUT

	# All the information you need for training.
	train:

	  # Where to get training data from.
	  # NOTE: `TRAIN_DATA` needs to have dictionary keys named `INPUT` and
	  #       `OUTPUT`, corresponding to the `INPUT` and `OUTPUT` names in the
	  #       model section above.
	  data:
		- pickle: TRAIN_DATA

	  # Try playing with the batch size and watching accuracy and speed.
	  provider:
	    batch_size: 32

	  # How many epochs to train for.
	  epochs: 10

	  # Where to load and save weights.
	  weights:
	    initial: INITIAL_WEIGHTS
		best: BEST_TRAINING_LOSS_WEIGHTS
		last: MOST_RECENT_WEIGHTS

	  # The optimizer to use. Try doubling or halving the learning rate.
	  optimizer:
	    name: adam
		learning_rate: 0.001

	# You need this section if you want to run validation checks during
	# training.
	validate:
	  data:
	    - pickle: VALIDATION_DATA

	  # Where to save the best validation weights.
	  weights: BEST_VALIDATION_LOSS_WEIGHTS

	# You need this section only if you want to run standalone test runs to
	# calculate loss.
	test:
	  data:
	    - pickle: TEST_DATA
	  # Which weights to use for testing.
	  weights: BEST_VALIDATION_LOSS_WEIGHTS

	# This section is for trying out your model on new data.
	evaluate:
	  # The data to supply as input. Unlike the train/validate/test sections,
	  # you do not need a corresponding `OUTPUT` key. But if you do supply one,
	  # Kur can save it to the output file for you so it's easy to play with
	  # during post-processing
	  data:
	    - pickle: NEW_DATA

	  # Which weights to use for evaluation.
	  weights: BEST_VALIDATION_LOSS_WEIGHTS

	  # Where to save the result (as a Python pickle)
	  destination: RESULTS.pkl

	# Required for training, validation and testing
	loss:
	  # You need an entry whose target is `OUTPUT` from the model section above.
	  - target: OUTPUT
	    
		# The name of the loss function. Change it if appropriate
	    name: categorical_crossentropy
	...

We're going to cover the simplest details of these sections.

- ``include``: You only need this if you've split your specification into
  multiple files. Otherwise, you can leave it empty or just remove it.
- ``settings``: This is the place that you can set global variables that you
  want to reference using the templating engine later (e.g., data sets or model
  hyperparameters). If you don't have any variables, you can just leave this
  section empty or remote it.
- ``model``: This is the fun part! Make sure you have an ``input`` entry, and
  a give the final layer a name, too (it's your output). The names need to
  correspond to the data that gets loaded during training, evaluation, etc.
  For a full list of "containers" (that's what Kur calls each entry in the model
  section), see :doc:`containers`. The :doc:`examples` are also a good place to
  start.
- ``train``: Everything you want to tell Kur about the desired training
  process.
  
	- The ``data`` section just tells Kur to load a pickled Python file called
	  ``TRAIN_DATA``. That file should be a Python dictionary with keys
	  corresponding to the input/output names you chose in the ``model``
	  section.  The values in that dictionary should be numpy arrays that you
	  want to feed into the Kur model.
	- The ``batch_size`` can be used to change how many training samples Kur
	  uses at each step in the training process.
	- ``epochs`` tells Kur how many iterations of the entire training set it
	  should run through before stopping.
	- The ``weights`` section tells Kur where it should save the state of the
	  model (the model *weights* or *parameters*). This section tells Kur to
	  load any existing weights from the ``initial`` file; these weights might
	  exist because you've already trained the model a few times and now you
	  want to train some more, picking up where you left off. If this
	  ``initial`` file doesn't exist, Kur just assumes it's your first time
	  through the training process and chugs along merrily. The ``best`` file
	  tells Kur where to save the weights if they produce the lowest loss (with
	  respect to the training data) that Kur has seen yet. The ``last`` file is
	  where Kur saves the weights before it stops training.
	- The ``optimizer`` is where you tell Kur which algorithm it should use to
	  try and improve the model performance / minimize loss.

- ``validate``: It's usually a good idea to have a validation set that you can
  use to independently assess how the model is performing. This is the place for
  it! It accepts a ``data`` section just like ``train``, and the ``weights``
  tell Kur where to save the weights whenever they produce the lowest historical
  loss with respect to the validation data.
- ``test``: If you have a test set, put its ``data`` specification here. The
  ``weights`` field tells Kur which weights you want it to load first.
- ``evaluate``: This is where you put information about new data sets you want
  to apply your model to. And you guessed it---the ``data`` section is just like
  all the others. The difference is that the pickled data dictionary not longer
  required a key corresponding to the model output from the ``model`` section;
  but if you give Kur the true output data anyway, it can use it for additional
  statistics and save it to the output for for you. The ``weights`` field tells
  Kur which weights to load before evaluating. ``destination`` names the output
  file where Kur should save the model results. It will save them as a Python
  pickle.
- ``loss``: Every model output needs a corresponding loss function. Make sure
  you have a ``target`` for each model output (it should have the same name,
  too, just like the data files).

Running Kur
===========

Kur is pretty easy to run. It looks like this:

.. code-block:: bash

	$ kur [-v] {train | test | evaluate | build} SPECIFICATION.yml

You give it your specification file, and you tell it what to do with the file.
By this point, train, test, and evaluate should make sense. The other command,
``build`` is kind of like ``train``, but does not load the data set, and doesn't
actually start training. Instead, it just assembles the model, which is useful
for checking for obvious problems.

By default, Kur only prints out warnings and errors to the console. You should
definitely pay attention to any warnings: they are indicative of something
unexpected or wrong. If you are curious about the lower-level details of what
Kur is doing, though, feel free to add ``-v`` to enable INFO-level output, or
even ``-vv`` to enable DEBUG-level ouput (there's a lot).

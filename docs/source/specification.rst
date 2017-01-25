*********************
Kurfile Specification
*********************

This is the more in-depth version of the :doc:`getting_started` page. It is
meant to detail the more advanced use and features of Kur specification files.
There will be some redundancy between the two pages, but this should be
considered the comprehensive overview.

Kur uses "specification" files to describe the model, hyperparameters, data
sets, training/evaluation options, and functional settings. Here, we describe
the required sections that every Kur model needs.

.. note::

	Out of the box, Kur understands YAML files, although other formats may be
	added in the future. For details of the YAML syntax, take a look at the
	`Ansible overview <https://docs.ansible.com/ansible/YAMLSyntax.html>`_.
	Here, we will give our examples in YAML, since the concepts should port
	fairly directly to other data languages (like JSON).

Top-level Sections
==================

All Kur specifications support these top-level sections:

.. code-block:: yaml

	---

	# Optional
	include:

	# Optional
	settings:

	# Required
	model:

	# Required for respective functionality.
	train:
	validate:
	test:
	evaluate:

	# Required for training, validation, and testing
	loss:

	...

The ``model`` section is required. The ``train``, ``test``, and ``evaluate``
blocks are required to train, test, or evaluate a model, respectively. The
``validate`` block is optional, and is only used during training; if present,
it describes the validation set to use during training.

Each of these sections is described in detail below.

.. _model_spec:

Model
=====

The model section is the most important part of the specification, and is the
only strictly-required section. It is where you get to describe the underlying
deep learning model that you want to use.

The model must contain a list of components, which Kur calls "containers"
(because they "contain" implicit tensor operations inside). The most general
container looks like this:

.. code-block:: yaml

	- CONTAINER_TYPE:
	
	    # Parameters that are given to the container.
	    param1: value1
	    param2: value2

	  # Optional information about this container.
	  name: NAME
	  tag: TAGS
	  oldest: OLDEST
	  inputs: INPUTS
	  sink: SINK

There are many different kinds of containers, which we describe on the
:doc:`containers` page. The parameters that are given to the container (e.g.,
``param1: value1``) are container-specific.

.. note::

	Kur actually thinks about two broad types of containers: *layers* and
	*operators*. Layers are containers which "know" the kind of underlying
	tensor operations they represent; think of layers as your deep learning
	primitives that you use to construct your model. Convolutions,
	fully-connected layers, recurrent layers, and so forth are all "layers" to
	Kur. Operators are "meta-layers." They modify the way other layers are
	interpretted, but do not produce underlying tensor operations themselves.
	For example, a "for" loop is an operator because it can generate many other
	layers, but by itself, a "for" loop is not a deep learning operation. If
	you've looked through the :ref:`in_depth_examples` page, then you know that
	``convolution``, ``activation``, ``dense``, and ``input`` are all layers.
	The ``for`` loop is an operator.

The other parameters to the container are described below.

Name
----

The ``name`` field gives the container a name that can be referenced by other
layers. If you don't need to reference this layer in your model, you probably
don't need to give it an explicit name. Simple models will probably only have
two named layers: the first layer (the input layer) and the last layer (the
output layer).  More complex layers might have multiple inputs, multiple
outputs, or more complicated, branched connections between containers, and then
naming your containers becomes
a lot more convenient.

Note that internally, all containers have a unique name: if you don't specify
one, Kur generates its own. Kur's generated names start with
double-underscores: ``__``. So if you are authoring your own containers, you
can rely on consistent and fully-defined names.

Names are unique and immutable. If you have two containers with the same name,
Kur will complain and ask you to fix it.

Example:

.. code-block:: yaml

	# Names are just strings.
	name: my_container_name

Tag
---

The ``tag`` field is kind of like a mutable name. It can be used to let a
container temporarily "nickname" itself. It looks like this:

.. code-block:: yaml

	# Single tag
	tag: foo

	# Multiple tags (short version)
	tag: [foo, bar, baz]

	# Multiple tags (long version)
	tag:
	  - foo
	  - bar
	  - baz

Now other layers can refer to that layer using the templating engine:
``{{ tags.foo }}`` and ``{{ tags["foo"] }}`` both resolve to the tagged
container's name.

Why do you need this? Well, names are immutable in Kur, but sometimes it's just
convenient to be able to temporarily name a container. If you don't know why
you'd use one, then you probably don't need one. Here's a longer snippet of how
a tag can be used:

.. code-block:: yaml

	# Create a layer and tag it.
	- convolution:
	    # ...
	  tag: foobar

	# ... more layers

	# Reference the tag.
	- convolution:
	    # ...
	  inputs: "{{ tags.foobar }}"

	# ... more layers

	# Reassign the tag.
	- dense:
	    size: 10
	  tag: foobar

The ``{{ tags.foobar }}`` in this example just resolves to the name of the
first convolution container. It is similar to this code, which does not use
tags.

.. code-block:: yaml

	- convolution:
	    # ...
	  name: my_convolution

	# ... more layers

	- convolution:
	    # ...
	  inputs: my_convolution

Tags are most useful in large, complicated models with many loops where you
might want to grab a container you created earlier, do something with it (e.g.
use it as input), but then you want to tag the new container with the same name
(e.g., for grabbing the new container next time through your loop).

Oldest
------

The ``oldest`` are like sticky tags: they are immutable (like names), but
reuseable (like tags).  Basically, oldest tags will always refer to the first
container to use the ``oldest`` tags, no matter how many other containers try
to claim that oldest tag in the future. ``oldest`` tags are declared just like
regular tags:

.. code-block:: yaml

	# Single "oldest" tag
	oldest: foo

	# Claim multiple "oldest" tags (short version)
	oldest: [foo, bar, baz]

	# Claim multiple "oldest" tags (long version)
	oldest:
	  - foo
	  - bar
	  - baz

They are also used in a similar way to regular tags:

.. code-block:: yaml

	- convolution:
	    # ...
	  oldest: foobar
	  name: first_convolution

	- convolution:
	    # ...
	  oldest: foobar
	  name: second_convolution

	- convolution:
	    # ...
	  oldest: [foobar, baz]
	  name: third_convolution

	# ... more layers

	# This convolution will get its input from `first_convolution`
	- convolution:
	    # ...
	  inputs: "{{ oldest.foobar }}"

	# This convolution will get its input from `third_convolution`
	- convolution:
	    # ...
	  inputs: "{{ oldest.baz }}"

Again, these ``{{ oldest.foobar }}`` variables just resolve to the names of the
referenced containers (e.g., ``first_convolution``).

Inputs
------

The ``inputs`` field specifies which containers this container should expect to
receive input from. Normally, a container's input is implicitly the most
recently declared container in the model. But sometimes when you have a more
complicated model (e.g., one with multiple inputs or with branching), you need
to be able to override this default Kur behavior and specify the input
containers manually.

The ``inputs`` field can be the name of a single container, or a list of names.
For example

.. code-block:: yaml

	# Single input
	inputs: my_layer

	# Multiple inputs (short version)
	inputs: [my_layer, your_layer]

	# Multiple inputs (long version)
	inputs:
	  - my_layer
	  - your_layer

Sink
----

Normally, a model's output containers are the last, unconnected containers in
the Kurfile, or standalone ``output`` layers. But Kur also allows you to
quickly tag a layer as an output layer without creating another layer entry.
You can do this by setting the ``sink`` field to a boolean true value (in YAML
you can do this with ``sink: [yes | true]``).

For example, consider this:

.. code-block:: yaml

	- convolution:
	    # ...
	  sink: yes
	  name: layer_1

	- convolution:
	    # ...
	  name: layer_2

The container ``layer_1`` is one of the model outputs. It is also an input to
``layer_2``. (Why? Because ``layer_2`` didn't declare an explicit ``inputs``,
so it still gets its input from the most recently declared container.) And if
``layer_2`` is the last layer in the model, then model will have a second
output named ``layer_2``.

Settings
========

The ``settings`` section is a place to declare global variables,
hyperparameters, and configure the Kur backend. It is an optional section, and
there are no required components of ``settings`` even if you do use it (i.e.,
it can be empty).

Let's talk about some of the things you can do with it.

Setting the Backend
-------------------

The Kur backend can be chosen like this:

.. code-block:: yaml

	settings:

	  backend:
	    name: NAME
	    variant: VARIANT
	    device: DEVICE
	    PARAM_KEY: PARAM_VALUE
	    PARAM_KEY: PARAM_VALUE
	    ...

The ``NAME``, ``VARIANT``, ``DEVICE``, and ``PARAM_`` fields are all optional.

The ``NAME`` field specifies which backend Kur should use (e.g., ``keras``). If
no ``NAME`` is specified (or indeed, if the entire ``backend`` or ``settings``
sections are absent), then Kur will attempt to use the first backend that is
installed on the system.

The ``VARIANT`` field takes a string or a list of strings that should be passed
to the backend. They do not have any defined meaning. They are useful for
developers who want to be able to make small, functional changes to an existing
backend without having to re-write an entire backend.

The ``DEVICE`` field tells Kur that it should use a particular device for its
calculations. This can be ``cpu``, ``gpu``, or ``gpuX`` where ``X`` is an
integer.

The remaining ``PARAM_KEY``, ``PARAM_VALUE`` fields are just key/value pairs
that the backend uses to configure itself. Their meaning is backend specific.

An example ``backend`` specification that asks Kur to use Keras over TensorFlow
is:

.. code-block:: yaml

	settings:
	  backend:
	    name: keras
	    backend: tensorflow

Global variables
----------------

The ``settings`` section is also a good place to put global variables. The
:ref:`CIFAR-10 example <in_depth_cifar_10>` is a good example of this, where the dataset
is defined once, and then referenced by other sections. In that example, YAML
language features (anchors and aliases) are used to reference the dataset.

The special thing about the ``settings`` section that makes it particularly good
for putting variables is that all of data loaded in the ``settings`` section is
available to all other sections through the templating engine. That means you
can do things like:

.. code-block:: yaml

	settings:
	  batch_size: 32

	train:
	  provider:
	    batch_size: "{{ batch_size }}"

Hyperparameters
---------------

For the same reason that the ``settings`` section is a good place for global
variables, it is also the best place for hyperparameters. Basically, treat your
hyperparameters like global variables, and reference them in your model. See the
:ref:`CIFAR-10 example <in_depth_cifar_10>` for a good use of this.

Include
=======

The ``include`` section is optional and lists one or more other specification
files that should be loaded and parsed alongside the current file. They are a
convenient way to separate dependencies or to split complicated configurations
into multiple files.

There are a couple ways to specify includes

.. code-block:: yaml

	# Include a single other file.
	include: other-file.yml

	# Include a single other file (list-of-files)
	include:
	  - other-file.yml

	# Include a single other file (list-of-dictionaries)
	include:
	  - source: other-file.yml

	# Include two other files (list-of-files, short version)
	include: [A-file.yml, B-file.yml]

	# Include two other files (list-of-files, long version)
	include:
	  - A-file.yml
	  - B-file.yml
	
	# Include two other files (list-of-dictionaries)
	include:
	  - source: A-file.yml
	  - source: B-file.yml

The ``include`` field is the very first field parsed out of every file. Each
include is parsed in order, recursively.

Now, you might ask: how does including actually work? Great question. Merging
complex data structures (like dictionaries of lists of dictionaries of ...) is
non-obvious. The best way to conceptualize this is to think of the YAML as just
a big data structure full of dictionaries, lists, and some primitives (like
integers). When you ``include`` a second file, the current specification file
gets merged into the content of the second include file (recursively). Keep
this in mind as you read through the different merging strategies that Kur
supports:

- ``blend``: This is the default strategy. Basically, all dictionaries
  (remember, at top-level, all specification files are just dictionaries) are
  merged by looking at their keys. If only one of the dictionaries has the key,
  then the key and value are kept in the merged result. If both dictionaries
  have the key, then:

	- If the data types of the values are *different* or if the data types are
	  *primitive* (integer, float, boolean), the "not included" dictionary's
	  value is kept (i.e., "includes" get overridden by the file doing the
	  including).
	- If the values are both dictionaries, they are recursively merged with the
	  same ``blend`` strategy.
	- If the values are both lists, then the two lists are merged into a single
	  list. Each element of the list is the resulting of ``blend``-ing the
	  corresponding elements of the two original lists. If one list is longer
	  than the other, then the "unmatched" elements are appended to the end of
	  the merged list (and are unaffected by the presence of the other list).

- ``merge``: This is similar to the ``blend`` strategy, except that lists are
  not merged, and are instead replaced as if they were primitives. Thus, the
  "not included" list is kept, overridding the include.
- ``concat``: This is also similar to the ``blend`` strategy, but instead of
  replacing or blending lists, they are simply concatenated. The "included"
  list is first, followed by the list from the "not included" source.

If you want to choose a strategy other than the default ``blend`` method, you
can do so using the list-of-dictionaries format:

.. code-block:: yaml

	# Include a single other file with an alternative merging strategy.
	include:
	  - source: other-file.yml
	    method: merge

	# Include two files, one with a non-default merge strategy
	include:
	  - source: A-file.yml
	    method: merge
	  - source: B-file.yml

Train
=====

The ``train`` section tells Kur how it should train your model: where the data
comes from, how many epochs it should train for, where it should save model
weights, where the log files are, etc. This section is required if you intend to
train a model, but is unnecessary if you are only testing or evaluating an
existing model. It looks like this:

.. code-block:: yaml

	train:

	  # How to load and process data (required)
	  data: DATA
	  provider: PROVIDER

	  # Where the log file lives
	  log: LOG (optional)

	  # How many epochs to train for (optional)
	  epochs: EPOCHS

	  # Where to store weights (optional)
	  weights: WEIGHTS

	  # What optimizer to use (optional)
	  optimizer: OPTIMIZER

	  # Callbacks to be executed after each epoch (optional)
	  hooks: HOOKS

The ``data`` and ``provider`` fields are discussed in the :ref:`data_spec`
section, and the ``hooks`` field is discussed in :ref:`hooks_spec`. The other
fields we discuss below.

.. _log_spec:

Log
---

The ``log`` field indicates where the log file should be stored and what format
it should be stored in. It is an optional field; if it is not specified, not log
file is saved or loaded.

What is saved in the log? The log contains statistics from the training process,
such as the loss from each model output. Because Kur stores loss values in the
log, it knows what the historically lowest loss values have been. As you will
see in the :ref:`weights_train` section, Kur can save the model weights which
have the lowest historical loss values. Kur will take into account loss values
from the logs when deciding if the current loss is, in fact, the lowest, *even
between independent training runs*.

Here are some examples of using this field:

.. code-block:: yaml

	# Empty entry: same as not specifying a log (no log will be used)
	log:

	# Explicitly empty entry: same as not specifying a log (no log will be used)
	log: null

	# Use the default log format
	log: /my/log/path

	# Use the default log format (alternative format)
	log:
	  path: /my/log/path
	
	# Non-default log format, optionally with implementation-specific parameters
	log:
	  name: LOGGER_TYPE

	  # Parameters to LOGGER_TYPE (e.g., `path`)
	  param: value
	  param2: value2

The default logger is a binary logger that saves log information in a binary
format, which allows data to be appended efficiently rather than spend precious
training time parsing complex formats before writing log data to disk (see
:ref:`this example <using_binary_logger>` of loading this file format).

Available loggers:

- ``binary``: the default binary logger. It creates an entire directory
  structure at ``path`` to store its statistics.

Epochs
------

The ``epochs`` field is an integer that simply tells Kur how many epochs to
train for during a ``kur train`` run. If it isn't specified (or if it is set to
an empty or null value), then Kur trains interminably (or rather, until you
Ctrl+C the process).

The ``epochs`` field tells Kur how many epochs to train for during a ``kur
train`` run. If it isn't specified (or if it is set to an empty or null value),
then Kur trains interminably (or rather, until you Ctrl+C the process). If you
set it to an integer, then Kur will train for that many epochs every time ``kur
train`` is called. More complicated configurations can be specified with:

.. code-block:: yaml

	epochs:
	  number: NUMBER
	  mode: MODE

``NUMBER`` is the number of epochs to train for. To train forever, set this to
``null`` or ``infinite``. For finite values of ``NUMBER``, ``MODE`` tells Kur
how to interpret ``NUMBER`` and can be one of the following:

- ``additional``. Kur will train for ``NUMBER`` epochs every time ``kur train``
  is called. This is the default, and is equivalent to the shorter ``epochs:
  NUMBER`` syntax.
- ``total``. Using the :ref:`log_spec`, Kur will train for exactly ``NUMBER``
  epochs total, regardless of how many times ``kur train`` is called. For
  example, let's say that ``NUMBER`` is 10 in ``total`` mode. You call ``kur
  train`` but interrupt it after epoch 6 completes. If you can ``kur train``
  again, it will only train for 4 more epochs (to reach its total of 10). If
  you call ``kur train`` a third time, it will simply report that has already
  finished 10 epochs. If a log is not specified, Kur will warn you but proceed
  training as if ``MODE`` were ``additional``.

Optimizer
---------

The whole point of training a model is to adjust the weights to minimize the
loss function. Deciding exactly how to adjust the weights is actually hard, and
it's called "optimization." Kur allows you to select an optimizer function for
training like this:

.. code-block:: yaml

	# Set the optimizer and use its default parameter values.
	optimizer: NAME
	
	# Set the optimizer, and optionally provide parameter values
	optimizer:
	  name: NAME

	  # Optional parameters
	  param: value

Available optimizers:

- ``adam``: The `Adam optimizer <arxiv.org/abs/1412.6980>`_. It takes these
  parameters:

    - ``learning_rate`` (default: 0.001). The learning rate for the optimizer.

- ``sgd``. Stochastic gradient descent. It takes these parameters:

	- ``learning_rate`` (default: 0.01). The learning rate for the optimizer.
	- ``momentum`` (default: 0)
	- ``decay`` (default: 0)
	- ``nesterov`` (default: ``no``). If True, Nesterov momentum calculations
	  are used.

- ``rmsprop``. RMSProp. It takes these parameters:

	- ``learning_rate`` (default: 0.001). The learning rate for the optimizer.
	- ``rho`` (default: 0.9)
	- ``epsilon`` (default: ``1e-8``)
	- ``decay`` (default: 0)

Additionally, all of these optimizers support these paramters:

- ``clip`` (default: ``null``). Scale or clip gradients. To scale the gradients
  so that their L2 norm never exceeds some value ``X``, use:

	.. code-block:: yaml

	    clip:
	      norm: X

  To clip gradients so that none of their absolute values exceeds ``X``, use:

	.. code-block:: yaml

	    clip:
	      abs: X

If no optimizer is specified, or if the name is mising, the ``adam`` optimizer
is used.

.. _weights_train:

Weights
-------

The ``weights`` section tells Kur where to load/save weights on disk. This
is important so that you can use the weights in the future (e.g., on a future
evaluation, or continued training, or even transfer learning).

If the ``weights`` section is missing, no weights will be loaded or saved, or
you could specify null weights like this:

.. code-block:: yaml

	# These are both the same as not loading or saving weights.
	weights:
	weights: null

You can also just specify a file name. This tells Kur to try and load initial
weights from the given path if the path exists. If the path doesn't exist, Kur
just keeps on going. Moreover, if you do *not* specify a ``weights`` field in
the :ref:`validate_spec` section, then Kur will use this path to save the best
model weights (the weights corresponding to the lowest loss during training).
This format looks like this:

.. code-block:: yaml

	# This loads its initial weights from `PATH`. If `PATH` doesn't exist, then
	# training continues anyway with fresh weights. If no weights are specified
	# in the ``validate`` section, then the very best training weights are saved
	# to `PATH`.
	weights: PATH

The most flexibility can be gleaned from a dictionary-like value:

.. code-block:: yaml

	# This format allows for more flexibility.
	weights:
	  # Load the initial weights from this path
	  initial: INITIAL

	  # If true/yes, then Kur will refuse to train unless INITIAL exists.
	  # By default, this is no/false.
	  must_exist: [yes | true | no | false]

	  # Where to save the best weights (with respect to training set loss).
	  best: BEST

	  # Where to save the most recent model weights.
	  last: LAST

Each of the fields is optional.

The best weights that Kur saves (whether specified with ``best:`` or just with
``weights: PATH``) are always the weights corresponding to the historically
lowest loss values. Kur uses its log, when available, to decide when it has
encountered a historically low loss value, even if it encountered it during a
previous training run. See :ref:`log_spec` for more information on saving to a
log.

.. _validate_spec:

Validate
========

The ``validate`` section tells Kur how it should validate your model. Validating
a model involves showing it a different data set during training to see how it
performs, and is used to judge how well the model is converging, cehck if it is
overtraining, and tune model hyperparameters. This section is ignored if Kur
is not training, and even then is still optional. The ``validate`` section looks
like this:

.. code-block:: yaml

	validate:

	  # How to load and process data (required)
	  data: DATA
	  provider: PROVIDER

	  # Where to store weights (optional)
	  weights: WEIGHTS

	  # Hooks for running some quick analysis on validation data between
	  # epochs (optional).
	  hooks: HOOKS

The ``data`` and ``provider`` fields are discussed in the :ref:`data_spec`
section, and the ``hooks`` field is discussed in :ref:`hooks_spec`. The other
fields we discuss below.

Weights
-------

The ``weights`` section is similar to the :ref:`weights_train` section for
training, and is optional. However, it only specifies one thing: where to store
the best model weights with respect to the validation loss (i.e., the model
weights which have historically yielded the lowest values of the loss function
when the model was evaluated on the validation set). Just as with the best
training weights, Kur uses the :ref:`log files <log_spec>` to decide when it
has encountered a historically low loss value.

These are all valid:

.. code-block:: yaml

	# Don't save weights based on the validation loss.
	# These two examples are the same as if the ``weights`` section was not even
	# present in the specification.
	weights: 
	weights: null

	# Save the best validation weights to `PATH`:
	weights: PATH

	# Same thing:
	weights:
	  best: PATH

Test
====

The ``test`` section tells Kur how it should test your model when ``kur test``
is used. Testing is used to assess model performance as a final step, after all
hyperparameter tuning is complete. Testing is a sacred process, since you don't
want to tune yor model against the test set; you just want to evaluate its
performance when, e.g., publishing/posting results. Functionally, it is very
similar to validation in that a data set is evaluted to determine its loss and
accuracy, but does not impact the model weights (i.e., it is not a training
process). This section is optional, and only needed if you want to run ``kur
test``. Unsurprisingly, the ``test`` section just needs data:

.. code-block:: yaml

	test:

	  # How to load and process data (required)
	  data: DATA
	  provider: PROVIDER

	  # Hooks for running some quick analysis on the model outputs (optional).
	  hooks: HOOKS

The ``data`` and ``provider`` fields are discussed in the :ref:`data_spec`
section, and the ``hooks`` field is discussed in :ref:`hooks_spec`.

Evaluate
========

The ``evaluate`` section tells Kur how it should evaluate your model.
Evaluation, often called prediction, is the process of applying a previously
trained model to new data and producing outputs that you intend to use. For
example, if you train an image recognition pipeline, then you want to evaluate
whenever you want to use the model in the real world to produce image classes
for new data. This section is only required if you want to run ``kur
evaluate``.

Unlike training, validation, and testing data sets, evaluation does not require
that its data providers supply "ground truth" information. However, if ground
truth is provided, then it can still use it to help you better assess accuracy
metrics or for post-processing.

The evaluation section looks like this:

.. code-block:: yaml

	evaluate:

	  # How to load and process data (required)
	  data: DATA
	  provider: PROVIDER

	  # Where to load weights from
	  weights: WEIGHTS

	  # The post-evaluation functions to apply.
	  hooks: HOOKS

	  # Where to store the final, evaluated results
	  destination: DESTINATION

The ``data`` and ``provider`` fields are discussed in the :ref:`data_spec`
section, and the ``hooks`` field is discussed in :ref:`hooks_spec`. The other
fields we discuss below.

Weights
-------

The ``weights`` section is similar to the :ref:`weights_train` section for
training. However, it only specifies one thing: where to load the model weights
from before evaluating. Technically, this is optional, but unless you give your
model previously trained weights, it will produce garbage outputs.

These are all valid:

.. code-block:: yaml

	# Don't load any weights.
	# These two examples are the same as if the ``weights`` section was not even
	# present in the specification.
	weights: 
	weights: null

	# Load the weights from `PATH`.
	weights: PATH

	# Same thing:
	weights:
	  initial: PATH

.. _destination_spec:

Destination
-----------

The ``destination`` field is basically just a special hook. It is an ``output``
hook that will always be executed last. Since it is just an ``output`` hook, it
accepts the same arguments as an ``output`` hook. See :ref:`hooks_spec` for
more details.

.. note::

	Why is the ``destination`` hook special? Why not just use the existing
	``hooks`` take care of this? Remember that your specification might be
	included by other specifications. Once merged, you might have lots of
	hooks, but you probably only want one "final" output product written to
	disk. If this is not what you want, that's fine: just don't use
	``destination`` and use ``output`` hooks whenever is appropriate. But lots
	of users don't want that, so we offer ``destination`` as a convenience
	function.

Loss
====

The ``loss`` section is where you specify a loss function that is used during
training, validation, and testing (it is not required for evaluation). Every
model output needs a corresponding loss function defined. It looks like this:

.. code-block:: yaml

	loss:

	  - target: MODEL_OUTPUT_1
	    name: LOSS_FUNCTION
	    weight: WEIGHT
	    param_1: value_1
	    param_2: value_2

	  - target: MODEL_OUTPUT_2
	    # ... etc

There is one loss function per model output (``target``). The loss function are
in no particular order, although if you have multiple loss function associated
with the same ``target``, then only the last one is kept. The ``target`` value
(e.g., ``MODEL_OUTPUT_1``) is required and must match the name of a container
in the :ref:`model specification <model_spec>`. ``name`` is the name of the
loss function to use and is also required. ``weight`` is a floating-point
number that tells the optimizer how much weight to give to this particular
model output when determining the total loss; it is optional and defaults to
1.0. If the loss function takes any other parameters, they are also included
alongside everything else (e.g., ``param_1: value_1`` above).

Valid loss functions (choices for ``name``) are:

- ``categorical_crossentropy``: Categorical crossentropy loss, which is an
  appropriate loss function for 1-of-N classification tasks.
- ``mean_squared_error``: Mean-squared error, which calculates the average
  the squared distance between the model outputs and the ground truth vectors.
- ``ctc``: Connectionist temporal classification. The is a soft-alignment loss
  function appropriate for functions like automatic speech recognition (ASR).

Using CTC Loss
--------------

CTC loss takes several extra parameters: ``input_length``, ``output_length``,
and ``output``. Your specification should look like this:

.. code-block:: yaml

	- name: ctc
	  target: PREDICTED_TRANSCRIPTION
	  output: TRUE_TRANSCRIPTION
	  input_length: LENGTH_OF_PREDICTED_TRANSCRIPTION
	  output_length: LENGTH_OF_TRUE_TRANSCRIPTION
	  relative_to: AUTOSCALE_TARGET

Here is description of all these pieces:

- ``PREDICTED_TRANSCRIPTION``: this is the name of your *model's output layer*,
  once it has passed through a softmax classification. Your model's output
  should be of shape ``(TIMESTEPS, VOCABULARY_SIZE+1)``, where
  ``VOCABULARY_SIZE`` is the number of "words" in your vocabulary (the ``+1``
  is needed to accommodate the CTC blank character). The model output should
  thus be one-hot encoded "words". The model will learn to insert CTC blank
  characters into the model output until the length of the output is
  ``TIMESTEPS``. ``TIMESTEPS`` should always be at least as large as the
  maximum true transcription.
- ``LENGTH_OF_PREDICTED_TRANSCRIPTION``. This is the name of the *data source*
  which contains the number of timesteps in the model's output to consider
  during loss function calculations. It should be a tensor of shape
  ``(NUMBER_OF_SAMPLES, 1)``, where each value is an integer indicating the
  length of the data in the ``AUTOSCALE_TARGET`` data source. By default,
  ``AUTOSCALE_TARGET`` is set to the ``PREDICTED_TRANSCRIPTION`` (output)
  layer. In this case, if all of your model's input samples span the entire
  duration of the input timesteps, then this length is just a constant value,
  equal to the number of timesteps outputted in the *output layer*. If your
  data samples are of difference sizes, try zero-padding them and providing the
  appropriately scaled number of timesteps as the length. For example, let's
  say you have a maximum of 200 frames of audio per input sample, which you
  then pass through a network that ultimately shapes the output into 32-length
  outputs. If you have an audio sample of length 140 frames, then you should
  set the ``LENGTH_OF_PREDICTED_TRANSCRIPTION`` length to ``ceil((140 / 200) *
  32) = 23`` for that sample. For complex models, it can be non-trivial to
  calculate this scaled value. In that case, it is easier to use
  ``relative_to`` (see ``AUTOSCALE_TARGET`` below).
- ``LENGTH_OF_TRUE_TRANSCRIPTION``. This is the name of the *data source* which
  indicates the number of "words" in each ground-truth transcription. It should
  be a tensor of shape ``(NUMBER_OF_SAMPLES, 1)``, where each value is an
  integer indicating the number of "words" in the true transcription. So if you
  are creating a character-level transcription model and one of your
  ``TRUE_TRANSCRIPTION`` entries is "hello world", then the corresponding entry
  in ``LENGTH_OF_TRUE_TRANSCRIPTION`` should be 11 (one for each character,
  including the space).
- ``TRUE_TRANSCRIPTION``. The name of the *data source* which contains the true
  transcriptions for each sample. This should point to a tensor of shape
  ``(NUMBER_OF_SAMPLES, MAX_TRANSCRIPTION_LENGTH)``. Each sample should be a
  vector with sparse one-hot encodings of the correspond words. So for example,
  if you have a character-level transcription of "hello world", then you might
  encode this as ``[7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3, 0, 0, ..., 0,
  0]``, where the encoding shown here is ``{'a' : 0, 'b' : 1, ..., ' ' : 26}``.
  Note that you need to pad it out (here, with ``0``'s) so that the total
  length is the maximum transcript length you are training on. The CTC blank
  character will automatically be inserted as ``number_of_words``.
- ``AUTOSCALE_TARGET``. Frankly, it can be a pain to need to determine your
  ``LENGTH_OF_PREDICTED_TRANSCRIPTION`` values. Moreover, as you start
  prototyping new models, the last thing you want to deal with is updating your
  dataset to reflect how the shape of the output layer depends on the shape of
  the input layer. So Kur can do this for you! To do this, set
  ``LENGTH_OF_PREDICTED_TRANSCRIPTION`` to a dataset containing the lengths of
  each *input sample* (e.g., audio utterance), then set ``AUTOSCALE_TARGET`` to
  the name of the *input layer*. Kur will then determine the appropriately
  scaled length of the predicted transcriptions by calculating how the shape
  of the input samples changes between the ``AUTOSCALE_TARGET`` layer and the
  ``PREDICTED_TRANSCRIPTION`` layer, and transform the lengths of the
  ``LENGTH_OF_PREDICTED_TRANSCRIPTION`` values appropriately. If
  ``AUTOSCALE_TARGET`` is not specified, it is equivalent to setting
  ``AUTOSCALE_TARGET`` to the output layer (``PREDICTED_TRANSCRIPTION``).

Overall, you should make sure these constraints are satisfied:

- Your model's output layer (``PREDICTED_TRANSCRIPTION``) is softmax'd, and are
  2D tensors: for each timestep, your feature vector should be one longer than
  your vocabulary size (to accommodate the CTC blank character). The number of
  timesteps can easily be larger than the length of the transcriptions you are
  trying to predict.
- The maximum value of ``LENGTH_OF_PREDICTED_TRANSCRIPTION`` is the number of
  timesteps in your model's output (again, often this is larger than the length
  of the transcription you are trying to predict). If you use
  ``AUTOSCALE_TARGET``, then the maximum value should be the number of
  timesteps in the layer pointed to by the ``AUTOSCALE_TARGET``.
- The maximum value of ``LENGTH_OF_TRUE_TRANSCRIPTION`` is less than or equal
  to the number of timesteps in your model's output.

Also remember that you essentially set the CTC loss function's ``target`` to
your model's output (``PREDICTED_TRANSCRIPTION``), and then you are adding
three new inputs to your model (which need to be defined in the training set):
``LENGTH_OF_PREDICTED_TRANSCRIPTION``, ``LENGTH_OF_TRUE_TRANSCRIPTION``, and
``TRUE_TRANSCRIPTION``.

For example, imagine you have audio samples, each with exactly 200 frames which
you are using to do character-level transcription. The number of characters in
your longest transcription is 16. Your vocabulary is A-Z plus the "space"
character (27 "words" total). You model's input should be ``[200, X]``, where
``X`` is the number of features for each audio frame. Your model's output
should be ``[Y, 28]`` after being softmax'd, where ``Y`` is at least 16 (but
realistically might be 64). Let's say the model's output layer is ``output``.
You need to provide additional input data sources:

- ``transcription``. Each sample should be length 16, and should look like
  ``[ 0, 15, 15, 11, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]``: length 16, with
  values indicating the encoded transcription (here, the word "apply", where
  ``{'a' : 0, ...}``).
- ``transcription_length``. Each sample should be length 1, and should look
  like ``[ 5 ]``, where ``5`` corresponds to the length of the transcription
  (here, the length of "apply").
- ``input_length``. Each samples should be length 1, and should look like ``[
  20 ]``, where ``20`` is the number of timesteps of the model input, scaled to
  the size of the output layer (here, ``64 * (5 / 16)``).

Your CTC loss function would be:

.. code-block:: yaml

	- name: ctc
	  target: output
	  input_length: input_length
	  output_length: transcription_length
	  output: transcription

Alternatively, you could use ``AUTOSCALE_TARGET`` (the value of ``relative_to``)
in order to simplify your calculations. In this case, your ``input_length``
data source would be the lengths of the input audio (in our example, 200, so
the ``input_length`` data source would be: ``[ [200], [200], [200], ... ]``)
and your CTC loss function would look like:

.. code-block:: yaml

	- name: ctc
	  target: output
	  input_length: input_length
	  relative_to: input
	  output_length: transcription_length
	  output: transcription

.. _data_spec:

Data Specification
==================

All of the train, validate, test, and evaluate sections can accept a ``data``
and a ``provider`` field. These are pieces that tell Kur where it can find data,
and how it should provide the data to the training (*mutatis mutandis*) process.
We'll talk about both of these sections below.

Data
----

The ``data`` section specifies a list of *data suppliers*. Suppliers are Kur's
name for objects which can produce one or more named data sources. Each supplier
can optionally consume some number of supplier-specific parameters. Thus, a
``data`` section generally looks like this, where ``SUPPLIER_1``, etc. are the
names of the Kur suppliers.

.. code-block:: yaml

	data:

	  - SUPPLIER_1:
	      param_1: value_1
	      param_2: value_2
	      # ....

	  - SUPPLIER_2:
	      param_1: value_1
	      # ...

	  # ...

Valid suppliers are:

- ``mnist``: This supplier provides MNIST data for the
  :ref:`in_depth_mnist_example` example. It takes two parameters: ``images``
  and ``labels``, each of which, in turn, is a :ref:`package_specification`.

  The MNIST supplier also takes care of creating a one-hot representation of the
  labels as well as normalizing the images. The images are presented to the
  network as single channel images (i.e., they are 3D).

- ``cifar``: This supplier provides CIFAR data for the :ref:`in_depth_cifar_10`
  example. In addition to standard :ref:`package_specification`, you can also
  specify:

	- ``parts``: Which parts of the data set to load. CIFAR-10 splits the data
	  sets into 6 pieces, named: 1, 2, 3, 4, 5, and "test". If ``parts`` is not
	  specified, all six pieces are loaded by the supplier; otherwise, ``parts``
	  can be a single piece to load, or a list of pieces to load.

- ``pickle``: Loads a pickled Python data structure. The pickled file is
  expected to contain a dictionary whose keys are strings naming the respective
  containers in the model, and whose values are numpy arrays. The name of the
  file is expected as the only argument to ``pickle``: ``pickle: PATH``.

- ``numpy_dict``: Loads a pickled Numpy dictionary. These files are created by
  taking a Python dictionary whose keys a strings naming the data, and whose
  values are numpy arrays, and saving the dictionary with ``numpy.save``. The
  name of the file is expected as the only argument: ``numpy_dict: PATH``.

- ``csv``: This supplier loads CSV data. If you only give it a filename, then
  it will try to load a local file, and it assumes that the first row of the
  file is a header row. Alternatively, you can given it a dictionary of
  arguments. In addition to the standard :ref:`package_specification`, you
  can also use these parameters (all of which are optional):

  .. code-block:: yaml

	csv:
	  format:
	    delimiter: DELIMITER
		quote: QUOTE_CHARACTER
	  header: HEADER
	  # ... also uses standard packaging

  ``DELIMITER`` is the delimiter character. Normally, it is autodetected, but
  you can override it here. Similarly, the ``QUOTE_CHARACTER`` indicates the
  character that begins/ends quoted strings, and is usually autodetected. The
  ``HEADER`` value is a boolean (``yes`` / ``no``) which indicates whether or
  not the first row of the file is a header row. If true, the names of the
  columns are used as the names of the data sources (e.g., you can use them in
  your model). If false, the first row is treated like data, and corresponding
  data sources of the form ``column_X`` are generated (``X`` is zero-based).
  By default, ``HEADER`` is true.

  .. note::

	At the moment, all CSV data will be cast to floating-point numbers. This
	means that if strings are encountered, you will get errors.

- ``speech_recognition``. This supplier loads data appropriate for automatic
  speech recognition (ASR, also known as transcription). It takes the standard
  :ref:`package_specification`, in addition to these other optional parameters:

	- ``unpack``: bool (default: True). If set, and if the source file is
	  compressed (e.g., ``.tar.gz``), then Kur will first unpack the file
	  before using the dataset.
	- ``type``: str, either ``spec`` or ``mfcc`` (default: ``spec``).
	  Determines the type of audio features to present to the model, either
	  spectrograms (for ``spec``) or Mel-frequency cepstral coefficients
	  (``mfcc``).
	- ``max_duration``: float (default: None). Only keeps audio utterances that
	  are shorter than ``max_duration`` seconds; if unspecified or ``null``, it
	  keeps all utterances.
	- ``max_frequency``: float (default: None). Only keep frequency components
	  that are less than ``max_frequency`` Hertz; if unspecified or ``null``,
	  it keeps all frequencies.
	- ``vocab``: str, list, or None (default: None). The vocabulary to use in
	  preparing transcripts. If None, it auto-detects the vocabulary from the
	  dataset (**note**: this is *only* recommended for testing). If a string,
	  it is a JSON file containing a single JSON list; each element in the list
	  is treated as a case-insensitive vocabulary word. If a list, each element
	  of the list is treated as a case-insensitive word.

  The speech recognition supplier will produce the following data sources that
  you can use in your model:

	- ``utterance``. The audio signal itself.
	- ``utterance_length``. The number of frames in the audio signal.
	- ``transcript``. An integer-encoded transcript.
	- ``transcript_length``. The length of the corresponding transcript.
	- ``duration``. The length of the audio utterance, in seconds.

  The input file can be a file (which is extracted) or a directory. Kur will
  search for a JSON-Lines (JSONL) file, each line of which should be a JSON
  directionary with the following keys:

	- ``text``: the transcription.
	- ``duration_s``: the duration of the audio, in seconds.
	- ``uuid``: a unique value used to identify the audio.

  Next to the JSONL file should be a directory named ``audio`` where all of the
  audio sources are stored. Each filename should be of the form ``UUID.EXT``,
  where ``UUID`` is the corresponding UUID in the JSONL file, and ``EXT``
  should be an extension identifying the format of the audio. Kur currently
  accepts the following formats: ``wav``, ``mp3``, and ``flac``.

The most important thing to realize about data suppliers is that the name of
the data sources must correspond to the inputs and, for training and testing,
the outputs of the model. For example, MNIST has an explicit ``images`` and
``labels`` keys, corresponding to the model containers from the example. CIFAR
has implicit ``images`` and ``labels`` keys that it creates internally.
Similarly, if you create a Python pickle, then the keys in the pickled
dictionary must correspond to the names of the input and output containers in
the model.

.. _package_specification:

Standard Packaging
``````````````````

Many of the data suppliers accept a standard set of parameters to make things
convenient for you. These parameters are: ``url``, ``checksum``, and ``path``,
and are interpreted like this:

- If ``path`` is given but ``url`` is not, then Kur will use a local file or
  directory (whether or not directories are allowed depends on the data
  supplier). If ``checksum`` is given, Kur will check that the file's SHA-256
  hash matches.
- If ``url`` is given but ``path`` is not, then Kur will download the URL to
  the system's temporary directory. If ``checksum`` is specified, Kur will
  check that the file's SHA-256 hash matches.
- If both ``url`` and ``path`` are specified, then Kur will only download the
  file if it doesn't already exist at ``path`` (``path`` can be a file or
  directory) or if its checksum fails (if specified).

Provider
--------

Data can come from many different places, at different rates, with different
latencies, etc. Sometimes it is all present at once and fits nicely in memory.
But that's not always the case. Kur helps you handle these different situations
with its *data providers* (not to be confused with *data suppliers*). Providers
are responsible for handing data to the model during training or evaluation in
nice, organized batches, and possibly shuffling the data between epochs.

Providers are specified like this:

.. code-block:: yaml

	provider:
	  name: NAME
	  param_1: value_1
	  param_2: value_2
	  # ...

The name of the provider is given by the ``name`` field, and everything else is
given to the provider as parameters. Valid provider names are:

- ``batch_provider``: A simple provider that can shuffle data and which presents
  data to the model in fixed-size batches. (An exception to this is the very
  last batch every epoch; if the size of the data set is not evenly divisible by
  the batch size, then the last batch is allowed to be a little smaller.) It
  accepts the following parameters:

	- ``randomize``: A boolean value ``yes, true, no, false`` indicating whether
	  or not the data should be shuffled between epochs. By default, it is true.
	- ``batch_size``: The number of samples to provide in each batch. By
	  default, it is 32.
	- ``num_batches``: An integer indicating how many batches to provide each
	  epoch. This is mostly useful for test purposes on slower machines. If it
	  is larger than the number of batches available, then all the batches are
	  kept. By default, all batches are provided. Note that even this is set
	  less than the number of available batches, the batches will still be
	  shuffled from across the entire dataset if ``randomize`` is True (i.e.,
	  you will get ``num_batches`` of randomly chosen samples, not simply the
	  first *N* batches repeatedly).
	- ``sortagrad``: A string specifying a data source. As Baidu noted in their
	  `DeepSpeech paper <https://arxiv.org/abs/1512.02595>`_, models can train
	  better and more stably if, during the first epoch, training samples are
	  presented in order of increasing duration. If a data source is specified
	  here, then for the first epoch, data will be sorted by this data source.
	  Setting ``sortagrad: X`` is equivalent to ``sort_by: X`` with
	  ``shuffle_after: 1``.
	- ``sort_by``: A string specifying a data source. If specified, all data is
	  sorted by this data source before the first epoch. By default, no sorting
	  is done.
	- ``shuffle_after``: An integer indicating how many epochs to wait before
	  randomizing the dataset. By default, this is zero.
	- ``force_batch_size``: A boolean indicating whether or not the
	  ``batch_size`` should be strictly adhered to. If this is True, then any
	  data samples that do not fit cleanly into fixed-sized batches are simply
	  dropped for that epoch (if shuffling is enabled, then you will still see
	  all your data samples at some point). If this is False, then Kur will try
	  its best to use fixed-sized batches, but may occassionally return smaller
	  batches (particularly at the end of the epoch if the length of the
	  training set is not evenly divisible by the batch size).

If the ``provider`` section is not given, or if ``name`` is not specified, then
a ``batch_provider`` is created as a default provider.

.. _hooks_spec:

Hooks
-----

Hooks are an opportunity to filter, transform, print, and/or save the model's
output. They do something a little different depending on which section in
your Kurfile you add them to:

- ``train``: the hooks are called between each epoch and are given the current
  epoch just completed and the current loss. This is useful for hooking into
  callbacks that notify you of your model's training progress.
- ``validate``: the hooks are passed a single batch of model output after each
  validation run. This is useful for printing out some examples of your model's
  progress.
- ``test``: the hooks are passed a single batch of model output once the
  testing run is complete. Like the ``validate`` hooks, they are useful for
  printing out some examples of your model's progress.
- ``evaluate``: the hooks are passed *all* the data generated during the
  evaluation run. This is useful for printing examples of model output, but
  also for transforming your data into more useful on-disk formats (e.g, taking
  the ``argmax`` of one-hot outputs, so you don't need to do it later).

In all cases, the ``hooks`` section is a list of hooks. Each hook is
a function that is applied, in order, to the model output. So if you have two
hooks ``F`` and ``G``, and the model output is ``x``, then the final result
that will be produced is ``G(F(x))``, so to speak. The exception is for
``train`` hooks, where each hook is simply run in sequence with epoch number
and the current loss value: ``F(epoch, loss)``, ``G(epoch, loss)``.

When do you want hooks? Usually in two cases:

- **Decoding**. Sometimes the model output is not in the format that is most
  usable to the rest of your system. You can use a hook to post-process /
  manipulate the data right within Kur.
- **Analysis**. Again, sometimes it's really convenient to be able to generate
  additional statistics right within Kur, as seen in the :ref:`MNIST example
  <in_depth_mnist_example>`. This is a nice place to do it.

Hooks can take parameters as well. An example of using hooks is:

.. code-block:: yaml

	hooks:
	  - output:
	      path: /path/to/output.pkl
	      format: pickle
	  - custom_function:
	      param: value

Many of these hooks will be application specific, but these hooks are available
as part of Kur:

- ``mnist``: This is a analysis hook used in the MNIST example, and is not
  appropriate for use outside of that example. It is intended as an
  ``evaluate`` hook.
- ``output``: This is used for saving intermediate data products. This is done
  by the :ref:`destination_spec`, but can also be done as a hook, which is nice
  when you want to save the model output, apply some other hooks, and then let
  ``destination`` save the final product as well. It takes two parameters:

    - ``path``: the path to save the data to.
	- ``format``: the data format to save the data as. Supported formats are:

	  - ``pkl`` or ``pickle``: Python 3 pickle. This is the default if
	    ``format`` is not specified.

  This hook is primarily an ``evaluate`` hook.
- ``transcript``: This is useful for performing argmax-decoding of the ASR
  pipeline, effectively turning your model outputs into true transcriptions.
  This is intended as a ``test``/``validate`` hook.

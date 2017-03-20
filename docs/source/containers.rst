******************************
Containers: Layers & Operators
******************************

Containers are the building blocks of Kur models. Each entry in a Kur model is
a container, and containers may accept other containers as input. There are two
types of containers: *layers* and *operators*. "Layers" are Kur's
representation of deep learning primitives (like convolutions). "Operators" are
like "meta-layers": they are used to modify how other layers are interpretted
(like "for" loops).

Layers
======

Layers are your fundamental building blocks for complex deep learning models.

Dense
-----

**Description**. A fully-connected layer of nodes (affine transformation), in
which each node is connected to each node from the previous layer.

**Purpose**. Dense layers are very common for "mixing" information across the
entire tensor. So in image classification, you usually want one or more dense
layers following your convolutions. They are also the building blocks of
multi-layer perceptrons.

**Note**. Dense layers do not include activations. If you want a non-linearity
between successive dense layers, you must explicitly insert them.

**Usage**::

	dense: SIZE

or::

	dense:
	  size: SIZE

- ``SIZE``: an integer or list of integers. If a single integer, it indicates
  the number of nodes in this dense layer. If a list, it is treated as a series
  of dense layers, one for each entry in ``SIZE``.

Convolution (CNN)
-----------------

**Description**. A convolution is a locally-connected layer (as opposed to a
dense / fully-connected layer).

**Purpose**. Convolutions are useful for "smoothing" out information, and is
used to abstract lower-level features into a higher-level representation. These
are very common in image recognition, where groups of pixels are combined to
produce some "high-level" meaning, which in turn is fed into deeper layers.

**Usage**::

	convolution:
	  kernels: KERNELS
	  size: SIZE
	  strides: STRIDES
	  border: BORDER

- ``KERNELS``: the number of convolutional filters to apply.
- ``SIZE``: an integer (for 1-D convolutions) or a list of integers which
  specify the size of the receptive field of the convolution. For image
  classification, for example, this would be ``[width, height]``, in pixels,
  of the convolution.
- ``STRIDES``: an integer or list of integers which indicate the stride (step
  size) of the convolution. If this is an integer, the stride is applied in
  each dimension; if it is a list, it indicates the stride in each dimension.
  This is optional; its default value is `1`.
- ``BORDER``: one of "valid" or "same", which tells Kur how to treat the edges
  of the input. Its default value is "same".

Recurrent (RNN)
---------------

**Description**. A recurrent layer for learning sequences of data.

**Purpose**. Recurrent layers are used to learn sequences.

**Usage**::

	recurrent:
	  size: SIZE
	  type: {lstm | gru}
	  sequence: SEQUENCE
	  bidirectional: BIDIRECTIONAL
	  merge: {multiply | add | average | concat}
	  outer_activation: ACTIVATION

- ``SIZE``: the number of recurrent nodes in the layer. This is the number of
  features that are kept *per timestep*. If ``SEQUENCE`` is true, then there
  are as many outputs as inputs (because the number of timesteps doesn't
  change), and at each timestep, the feature vector has length ``SIZE``. If
  ``SEQUENCE`` is false, then the output of the RNN is length ``SIZE`` (because
  only the last timestep is kept).
- ``TYPE``: the type of the recurrent cells (defaults to ``gru``).
- ``SEQUENCE``: boolean. If true, returns the entire sequences of RNN outputs.
  If false, only the last element of the sequence is returned. Defaults to
  True.
- ``BIDIRECTIONAL``: boolean. If true, a bidirectional RNN is constructed (one
  which learns both the forward and backward sequences of data).
- ``MERGE``: if ``BIDIRECTIONAL`` is true, then this determines how the outputs
  of the forward and backward RNNs is merged. If bidirectional is not set, then
  "merge" cannot be used. The default value is ``concat``.
- ``ACTIVATION``: an activation to apply within the RNN (it is not equivalent
  to simply following the RNN layer with an Activation layer). Defaults to
  "relu".

Activation
----------

**Description**. An activation layers applies a simple non-linearity to each
element of the tensor.

**Purpose**. Activations are non-linear, so they are an important part of any
deep learning model; without them, complex operations could be reduced to
simple multiplications. You usually want them following your convolutions, and
you typically want them as a final layer in your model.

**Usage**::

	activation: {softmax | relu | tanh | sigmoid | none}

Output
------

**Description**. A convenience layer for explicitly marking a point in the model
as an output.

**Purpose**. Any layer can be marked as an output using the ``sink: True``
value. But this is often awkward, especially if preceding layers are nested
deeply within a ``for`` or ``parallel`` block. This can be used to resolve
those ambiguities, as well as make your model easier to read.

**Usage**::

	output: NAME

or::

	output:
	  name: NAME

- ``NAME``: the name of the output layer. It will be associated with the
  tensors produced by the most recent layers.

Pooling
-------

**Description**. A pooling layer is used to reduce the size of the model's
representation in a local manner.

**Purpose**. Pooling layers are useful for "shrinking" the model by only
keeping the "most important" local information in the tensor. You often use
them in a convolutional stack: convolutions can increase the size of the
representation if its ``kernels`` value is higher than the previous layers, and
so pooling layers can reduce the size again, keeping only the most important
local information. This reduces training time and overfitting.

**Usage**::

	pool:
	  size: SIZE
	  strides: STRIDES
	  type: {max | average}

- ``SIZE``: an integer (for 1-D pools) or a list of integers which specify the
  size of the pooling layer's receptive field. For image classification, for
  example, this is the ``[width, height]``, in pixels, of the pool.
- ``STRIDES``: an integer or list of integers which indicate the stride (step
  size) of the pool. If this is an integer, the stride is applied in each
  dimension; if it is a list, it indicates the stride in each dimension.  This
  is optional; its default value is `1`.

Batch Normalization
-------------------

**Description**. A layer which directs Kur to normalize activations across the
previous layer.

**Purpose**. Batch normalization is useful to help prevent bad initializations
and improve convergence.

**Usage**::

	batch_normalization

or::

	batch_normalization:
	  axis: AXIS

- ``AXIS``: an integer indicating which axis to apply the normalization to.
  Defaults to -1 (the last axis).

Expand
------

**Description**. An "expand" layer inserts a new axis into the tensor.

**Purpose**. An expand layer is useful for manipulating the shape of your data
or tensors to fit them into the rest of your model. For example, if you have
a bunch of ``[width, height]`` single-channel (monochrome) images, you can't
simply pass them into a two-dimensional convolution, since N-dimensional
convolutions expect (N+1)-dimensional data. So you can insert an expand layer
to make your image data have shape ``[width, height, 1]``. Three-channel (RGB)
color images are already this shape ``[width, height, 3]``, so they don't need
an expand layer.

**Usage**::

	expand: DIMENSION

or::

	expand:
	  dimension: DIMENSION

- ``DIMENSION``: the index to insert the new length-1 dimension at (zero-based
  index). So if ``DIMENSION`` is zero, it will reshape your data to be ``[1,
  ...]``. Negative numbers count from the back, so if ``DIMENSION`` is -1, your
  data will be shaped to be ``[..., 1]``.

Flatten
-------

**Description**. A flatten layer reduces the dimension of a tensor to 1-D.

**Purpose**. Flatten layers are used when you are no longer interested in the
dimensionality of your data, and are ready to let the data "mix." You typically
put them immediately before the first dense layer of your model, since dense
layers require 1-D data.

**Usage**::

	flatten

or::

	flatten:

Parallel
--------

**Description**. A parallel layer applies a series of tensor operations to each
top-level element of your data. This is like a `map function
<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_.

**Purpose**. Here are a few (equivalent) ways to think about parallel layers:

	- Map function. Parallel layers apply a function to each (top-level)
	  element of your data, converting *[x0, x1, x2, ...]* to *[f(x0), f(x1),
	  f(x2), ...]*. An ``activation`` layer applies a function to each
	  *bottom-level* element of your data (you might say that an ``activation``
	  layer is applied element-wise, in some sense) and only applies primitive
	  operations. A parallel layer can apply very sophisticated deep learning
	  operations to each element of your data.
	- Sub-model. You can think of parallel layers as sub-models or nested
	  models.  They are models inside the deep learning model itself. You get
	  to define your sub-model within the ``parallel`` layer, and then it gets
	  applied to each element of your data.
	- Distributed operations. If you think of your data as a time series, where
	  each element represents another time step, then a ``parallel`` layer
	  applies the *same* operation (with the *same* weights) to each time step.

**Usage**::

	parallel:
	  apply:
	    - CONTAINER_1
	    - CONTAINER_2
	    ...

The ``apply`` key is a list of layers (or operations) that define the
"parallel" / "sub-model" / "time-distributed" operation. Each container in the
``apply`` list is applied, in turn, to each element of the input data.

Reuse
-----

**Description**. A reuse layer is a weight-sharing layer; it simply re-applies
another layer in the model without declaring new weights.

**Purpose**. A reuse layer is useful when you want to have a single tensor
operation (with learnable weights) that you apply in multiple places in your
model (as opposed to multiple tensor operations in multiple places).

**Usage**::

	reuse:
	  target: TARGET

or::

	reuse: TARGET

- ``TARGET``: the name of the layer to re-apply.

Tranpose
--------

**Description**. A transpose layer shuffles the dimensions of the data tensor.

**Purpose**. Sometimes you need to reorder your dimensions so that downstream
layers can act on your data in different ways. The Transpose layer helps you do
this.

**Usage**::

	transpose: PERMUTATION

or::

	transpose:
	  axes: PERMUTATION
	  include_batch: INCLUDE_BATCH

- ``PERMUTATION``: a list of axes which specifies the permutation to apply to
  the data tensor. You must specify the complete permutation (one entry for
  each dimension in the data tensor). If ``INCLUDE_BATCH`` is False (the
  default), then the axes are zero-based, starting with the first dimension
  after the batch. If ``INCLUDE_BATCH`` is True, then the axes are zero-based,
  but start with the batch dimension itself.
- ``INCLUDE_BATCH``: a boolean which indicates whether or not to include the
  batch dimension in the permutation. By default, it is False. This should
  almost certainly remain False for most applications, since you probably don't
  want your layers to act differently depending on batch size!

Merge
-----

**Description**. A merge layer combines multiple inputs into a single tensor.

**Purpose**. If your model has branches (a single layer is used as input by
multiple layers) or multiple inputs, then at some point you may want to combine
information from different layers. The Merge layer takes multiple inputs and
produces a single output.

**Usage**::

	merge: MODE

or::

	merge:
	  mode: MODE
	  axis: AXIS

- ``MODE``: the function to use to combine input data. It can be one of these:

	- ``average``: Computes the mean of the input tensors.
	- ``multiply``: Computes the product of the input tensors.
	- ``add``: Computes the sum of the input tensors.
	- ``concat``: Concatenates the data tensors along ``AXIS`` axis. This is
	  the default.

- ``AXIS``: the axis to concatenate input tensors along. Only used if ``MODE``
  is ``concat``. Defaults to -1 (the last axis).

.. _ref_placeholder:

Placeholder
-----------

**Description**. A placeholder layer just declares an input to the model. They
do not do anything other than declare where the model "starts."

**Purpose**. There are two purposes for input layers. The first is to help with
authoring, so you can keep track of your model's inputs. The second is to help
with debugging: input layers can have explicit sizes associated with them, and
if you try to apply your model to data of a different size, you'll get an
error.

**Usage**::

	input: NAME

or::

	input:
	  shape: SHAPE

- ``NAME``. The name of the model input. It must match the name of one of the
  data sources in the model specification. The first form is just a convenience
  form; names can be specified in the second form using the standard
  specification, like this::

	input:
	  shape: SHAPE
	name: NAME

- ``SHAPE``: A list of dimensions describing the expected shape of the data.
  This is useful for catching data problems early (the model will refuse to
  train/evaluate if it is given data of the wrong shape), and you can use the
  ``kur build`` command to test that your model "fits together" if the shapes
  of all inputs are specified. However, the shape is optional; if omitted, it
  will be inferred from the data source.

Repeat
------

**Description**. Repeats an input vector a given number of times.

**Purpose**. Repeat layers are great for taken 1D input (like the output of a
``dense`` or ``flatten`` layer) and producing 2D output. It does this by simply
duplicating the input a fixed number of times, taking input ``X`` and producing
``[X, X, X, ..., X]``.

**Usage**::

	repeat: COUNT

or::

	repeat:
	  count: COUNT

- ``COUNT``: An integer. The number of times to repeat in input vector.

Embedding
---------

**Description**. A learnable embedding layer.

**Purpose**. Embedding layers learn to map integer labels onto a
higher-dimensional vocabulary.

**Usage**::

	embedding:
	  vocab_size: VOCAB_SIZE
	  size: SIZE

- ``VOCAB_SIZE``: the size of the input vocabulary. An embedding layer takes
  integer labels as inputs, corresponding to vocabulary words. Thus, the inputs
  should be zero-based indices which are < ``VOCAB_SIZE``.
- ``SIZE``: The dimensionality of the embedding layer to learn. Each input
  label is mapped to a ``SIZE``-dimensional vector.

Dropout
-------

**Description**. A dropout layer which drops a random set of activations on the
previous layer during every forward/backward pass.

**Purpose**. Dropout layers help to control overtraining by forcing the network
to use different "neural pathways" to solve the problem. It is similar to
generating a large number of superimposed models.

**Usage**::

	dropout: FRACTION

- ``FRACTION``: float between 0 and 1. The fraction of activations to drop at
  each pass.

Operators
=========

Operators are used to manipulate deep learning operations at a higher level.
They are used to build more general, parameterized models.

For Loop
--------

**Description**. A traditional ``for`` loop, which can create an arbitrary
number of layers.

**Purpose**. The ``for`` loop is perfect when you want some part of your model
to be repeated a fixed number of times, but you want to parameterize that
number (e.g., turn it into a variable instead of simply copy/pasting the layers
over and over again).

**Usage**::

	for:
	  range: RANGE
	  with_index: INDEX
	  iterate:
	    - CONTAINER_1
	    - CONTAINER_2
	    ...

- ``RANGE``: the number of times to iterate.
- ``INDEX``: the name of the local variable to create. This is optional and
  defaults to ``index``.

The ``for`` loop adds each container under ``iterate`` during each iteration
(there are ``RANGE`` iterations total). Unless the containers are ``reuse``
containers, each resulting container is independent and has its own weights.

Debug
-----

**Description**. A debug message printer.

**Purpose**. The debug layer does not affect the model in any way. It is used
solely for outputting information to the console to aid in debugging your
models. You can print out variable values, for example.

**Usage**::

	debug: MESSAGE

**Note**: you must have debug-level (``kur -vv ...``) log messages enabled to
see this.

Assert
------

*Description*. A debug-style assertion checker.

**Purpose**. The assert layer does not affect the model in any way, but simply
provides a debug-style "assert" condition that will immediately raise an
AssertionError if the condition evaluates to False. This can be useful for
checking that your model is not being constructed incorrectly and that it's shape
is correct.

**Usage**::

	assert: CONDITION

- ``CONDITION``: the condition to assert must be True. It will be evaluated by
  an Engine before being checked for true/false.

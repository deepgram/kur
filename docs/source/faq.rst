**************************
Frequently Asked Questions
**************************

Backends
========

How do I use other backends?
----------------------------

Kur supports Theano, TensorFlow, and PyTorch backends. The default PyPI
installation only installs the Theano backend, though. This is because Theano
supports both CPU and GPU out-of-the-box, and supports all of Kur's current
feature set (except for multi-GPU). However, if you want try other backends,
feel free!

- **TensorFlow**. To install a CPU-only version of TensorFlow, you can simply
  do: ``pip install tensorflow``. For GPU support as well, do: ``pip install
  tensorflow-gpu`` instead. To use the backend, merge these lines into your
  Kurfile:

  .. code-block:: yaml

    settings:
      backend:
        name: keras
        backend: tensorflow

- **PyTorch**. See the `PyTorch homepage <https://pytorch.org/>`_ for the
  latest instructions, as their pre-built Python "wheels" depend on your
  version of Python and, for GPU support, your version of CUDA. Then merge
  these lines into your Kurfile:

  .. code-block:: yaml

    settings:
      backend:
        name: pytorch

Why would I use other backends?
-------------------------------

Good question! Turns out, implementation differences can lead to drastic
performance differences between backends. This often depends on your model
architecture, though, so there isn't a simply "always use such-and-such a
backend" statement that can be made. So it is usually worth the time to try
different backends and see how well they work. Here are some overall
characteristics:

- **Theano**. This a great, overall performer. It is easy to install and
  therefore comes bundled in Kur as the default backend. It has two downsides:
  it does not support multiple GPUs, and its CPU implementation currently will
  only saturate a single core. If you only have one GPU, though, perfect! Note
  that we've seen more reports of NaN errors with Theano's RNNs than in other
  backends.

- **TensorFlow**. TensorFlow has a large backing from Google. It is slightly
  less obvious to install (because you need to pick a CPU-only vs. GPU
  package). However, it does do a great job at using all the CPUs at its
  disposal and supports multi-GPU.

- **PyTorch**. PyTorch is the new kid on the block. It is a well-designed,
  easy-to-use deep learning library. Someday, Kur may switch to PyTorch as the
  default backend. Right now, it will happily saturate many CPUs and can use
  multiple GPUs. There is no CTC loss function available at the moment, though,
  so you cannot train the speech recognition example using it.

Beyond these slight feature differences and the potential performance
differences, you might try different backends because it makes sense for the
rest of your development workflow. After all, the weight files saved and loaded
by Kur can also be saved and loaded into your favorite backend. That means that
you can port your current, written-by-hand TensorFlow model into Kur and keep
all your pre-trained weights! Or you can use Kur to explore the best models for
your problem, train for a while, and then export the weights so that you can
integrate them into your PyTorch model.

Releases
========

How do you do your versioning?
------------------------------

We use `Semantic Versioning <http://semver.org/>`_. It makes it really easy for
our users to know what the impact of updating will be. In a nutshell, semantic
versioning means that all of versions follow a ``X.Y.Z`` format, where:

	- X: major version. Changes whenever there are backwards-**incompatible**
	  API changes.
	- Y: minor version. Changes whenever there are backwards-**compatible**
	  feature changes.
	- Z: micro/patch version. Changes whenever there are bug fixes that do not
	  add features.

.. _why_python2:

Do you support Python 2?
------------------------

No, and we won't. Python 3 was released in 2008, and Python 2 was last released
in 2009. Python 3 is definitely the more modern, streamlined, convenient, and
(increasingly) supported language. Rather than make our code ugly with lots of
hacks (like using ``__future__``) or restricting ourselves to a common grammar
that both languages support, we are going to continue working in Python 3.

We will occassionally bend a little to include Python 3.4 support, but we don't
support earlier versions.

Tensor Shapes
=============

How are tensor dimensions ordered?
----------------------------------

For convolutions, we follow the same convention as TensorFlow, where the "color
channels" comes last:

	- 1D: ``(rows, channels)``
	- 2D: ``(rows, columns, channels)``
	- 3D: ``(rows, columns, frames, channels)``

For recurrent layers, we follow the same convention as Keras: ``(timesteps,
feature_dimensions)``.

Usage
=====

Why does Kur take so long to run?
---------------------------------

It doesn't. It's actually the compiler running in the background, something
that all deep learning libraries must do to increase performance. See
:ref:`this answer <looks_stuck>` for more information.

.. Kur documentation master file, created by
   sphinx-quickstart on Wed Nov 23 12:41:50 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |LICENSE| image:: https://img.shields.io/badge/license-Apache%202-blue.svg
   :target: https://github.com/deepgram/kur/blob/master/LICENSE

******************************
Kur: Descriptive Deep Learning
******************************

|LICENSE|

Contents:

.. toctree::
   :maxdepth: 1

   examples
   tutorial
   getting_started
   specification
   containers
   misc

Introduction
============

Welcome to Kur, the future of deep learning! Kur is the latest and greatest
deep learning system because:

- You can design, train, and evaluate models *without ever needing to code*.
- You *describe* your model with easily undestandable concepts, rather than
  trudge through *implementing* the model in some lower-level language.
- You can quickly iterate on newer and better versions of your model using
  easily defined hyperparameters and all the power of the `Jinja2
  <jinja.pocoo.org>`_ templating engine.
- **COMING SOON**: You can share your models (in whole or part) with the
  community, making it incredibly easy to collaborate on sophisticated models.

Go ahead and give it a whirl: :ref:`get_the_code` and then :ref:`Try It Out
<short_examples>`! Then build your own model in our :doc:`tutorial`.

What is Kur?
------------

Kur is a system for quickly building and applying state-of-the-art deep
learning models to new and exciting problems. Kur was designed to appeal to the
entire machine learning community, from novices to veterans. It uses
specification files that are simple to read and author, meaning that you can
get started building sophisticated models *without ever needing to code*. Even
so, Kur exposes a friendly and extensible API to support advanced deep learning
architectures or workflows. Excited? Jump straight into the
:ref:`examples <short_examples>`.

How is Kur Different?
---------------------

Kur represents a new paradigm for thinking about, building, and using state of
the art deep learning models. Rather than thinking about your architecture as a
series of tensor operations (``tanh(W * x + b)``) and getting lost in all the
details, you can focus on **describing** the architecture you want to
instantiate. Kur does the rest.

The Kur philosophy is that you should describe your model once and simply.
Simple doesn't mean brainless, nor does it imply that you are limited in what
you can do. By "simple" we mean that models should be simple to use, simply to
modify, and simple to share. A flexible, more general model is elegant. And
this makes it easier to reuse in new contexts or to share with the community.
Kur's power lies in quickly making models that are both flexible and reusable.

Aside: Brief History Lesson
---------------------------

Decades ago, researchers wrote low-level code using highly optimized linear
algebra libraries and ran the code on CPUs. After the rise of General Purpose
Computing on GPUs (GPGPU), researchers modified their code to use CUDA or
OpenCL. Although this code was functionally identical, GPU computing
represented an incredible breakthrough in efficiency, as these new machine
learning models could train and predict in fractions of the time compared to
CPUs. Problematically, these programs were relatively hard-coded; exploring
different hyperparameters or architectures typically required detailed
knowledge of the code, and was fraught with ugly and bug-prone hacks.

Eventually, computer scientists began abstracting away the low-level, dirty
details of highly-optimized CUDA code, and projects like `Theano
<deeplearning.net/software/theano/>`_ and `TensorFlow
<https://www.tensorflow.org/>`_ were born. These tools are incredible in that
they hide many of the implementation details of working with hardware (i.e.,
CPUs and GPUs), and instead expose higher-level tensor operations to the
developer. Even then, the developer is forced to choose between building up
higher-level abstractions of deep learning primitives, or devolving to the
rigid or hacked models of earlier years.  Projects like `Keras
<https://keras.io/>`_ and `Lasagna <https://github.com/Lasagne/Lasagne>`_
emerged organically, driven by a need to more quickly and intuitively develop
deep learning models. Their primary genius is in providing an API that maps to
the way people actually think about the components of a deep learning network
(e.g., as a "LSTM layer" rather than as a series of tensor operations).

Kur is the natural progression of these tools and abstrations. It allows you,
the researcher, to get straight to the heart of deep learning: develop that
awesome model you've been dreaming about in a few short lines. And best of all,
you craft your model with high-level abstractions rather than having to think
about annoying questions like:

- Which language should I use?
- Which backend is the best?
- What if I want to quickly test different model configurations?

.. _get_the_code:

Get the Code
============

At the moment, Kur includes Keras (on Theano) as a default backend. Installing
Kur will also install those dependencies. This makes it easy to start running
the examples.

From PyPI
---------

.. code-block:: bash

	$ pip install kur

From GitHub
-----------

Just check it out and run the setup script:

.. code-block:: bash

	$ git clone https://github.com/deepgram/kur
	$ cd kur
	$ python setup.py install

.. _short_examples:

Try It Out!
===========

Remember, there are more examples on the :doc:`examples` page.

MNIST: Handwriting recognition
------------------------------

.. include:: mnist-example.rst

Excited yet? Try tweaking the ``mnist.yml`` file, and then continue the
tutorial over on the :ref:`Examples <continue_examples>` page to see more
awesome stuff!

The YAML specification that Kur uses is described in more detail in the
:doc:`getting_started` page.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


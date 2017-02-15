***************
Troubleshooting
***************

Installation
============

I got a Python 2 error during installation. Now what?
-----------------------------------------------------

When installing Kur, you may encounter this error::

    ============================================================
    
                               ERROR
    
                 Kur requires Python 3.4 or later.
            See our troubleshooting page to get started:
    
     https://kur.deepgram.com/troubleshooting.html#installation
    
    ============================================================
    
    Uh, oh. There was an error. Look up there ^^^^ and you'll be
                training awesome models in no time!

This is because your version of Python is too old:

.. code-block:: bash

	$ python --version
	Python 2.7.12

Kur only supports Python 3.4 or higher (our :ref:`reasoning is here
<why_python2>`). If you thought you had Python 3 installed, then you should
:ref:`set up a virtual environment <virtualenv_setup>` to prevent your computer
from getting confused about which version of Python it should use. If you
create a ``virtualenv`` using the ``-p /usr/bin/python3`` option (as we show in
our :doc:`install` guide), then the virtual environment will automatically
use Python 3. Of course, you need to make sure you remember to :ref:`activate
the virtual environment <virtualenv_setup>` before running Kur!

Running Kur
===========

.. _looks_stuck:

Kur doesn't look like it is doing anything.
-------------------------------------------

Chances are, the model is still compiling. These deep learning models are
highly optimized in order to execute quickly. The backends are responsible for
calling low-level compilers (CUDA, GCC, etc.) in order to convert your Kur
model into something your processor or GPU knows how to use. For large models,
this can take a long time. Try running Kur with increased verbosity (``kur -v
...``) and see if, in fact, Kur says that it is waiting for the model to finish
compiling.

Kur takes a very long before it starts training.
------------------------------------------------

See :ref:`this answer <looks_stuck>`.

.. _theano_optimizer:

Theano is complaining about BLAS.
---------------------------------

Theano might throw an exception that looks like this::

	AssertionError: AbstractConv2d Theano optimization failed: there is no
	implementation available supporting the requested options. Did you exclude
	both "conv_dnn" and "conv_gemm" from the optimizer? If on GPU, is cuDNN
	available and does the GPU support it? If on CPU, do you have a BLAS
	library installed Theano can link against?

There are a number of ways to solve this.

- If you are trying to run on a CPU, do *one* of the following.

	- Switch another backend. For example, if you are currently using the Keras
	  backend with Theano (the default), try switching to the TensorFlow
	  backend:

	  .. code-block:: yaml

	      settings:
	        backend:
	          name: keras
	          backend: tensorflow

	- If you want to use the Keras backend with Theano, then you can add the
	  ``optimizer: no`` setting to your specification file:

	  .. code-block:: yaml

	    settings:
	      backend:
	        name: keras
	        backend: theano
	        optimizer: no

	- If you are using the Keras backend with Theano programmatically through
	  the Python API, you can pass the Keras backend constructor an additional
	  parameter:

	  .. code-block:: python

	    backend = KerasBacked(optimizer=False)

	- Install a linear algebra library. This depends a little on your platform.
	  For Ubuntu, you can do this:

	  .. code-block:: bash

	  	sudo apt-get install libblas-dev liblapack-dev gfortran-4.9

	- Disable optimizer in Theano globally. Edit your ``~/.theanorc`` file and
	  make sure these lines are present::

		[global]
		optimizer = None

- If you are trying to run on an NVIDIA GPU
	- Install cuDNN from NVIDIA's website.

Theano complains with error "Optimization failure due to: constant_folding"
---------------------------------------------------------------------------

See :ref:`this answer <theano_optimizer>`.

Theano complains about an ``ImportError: ... file too short``
-------------------------------------------------------------

This can be caused if the Theano cache becomes corrupt. Try moving (or, if you
are brave, deleting) the ``~/.theano`` directory someplace else and trying
again.

TensorFlow throws an ``ImportError`` about ``GLIBC`` not found.
---------------------------------------------------------------

This is an unfortunate problem in TensorFlow that boils down to this: your
operating system's version of ``libc`` (the C standard library) is too old. On
some platforms, you can easily upgrade the OS or ``libc`` and fix the problem;
on other platforms, it isn't as easy. For example, we've seen this bug crop up
on Ubuntu 12.04 (Precise Pangolin), but upgrading the distribution to Ubuntu
14.04 (Trusty Tahr) resolved the problem.

If you don't know how to upgrade your system, or if you just don't want to,
then the easiest workaround is to simply not use TensorFlow, and instead use a
backend based on, e.g., Theano instead. If you are using Keras as a backend to
Kur, then you can request that Keras use Theano behind the scenes by putting
this in your specification:

.. code-block:: yaml

	settings:
	  backend:
	    name: keras
	    backend: theano

Couldn't find ffmpeg or avconv
------------------------------

So, you want to do some speech-based learning. Great! In order to handle the
wide variety of audio and video formats that your training set might need, we
use ``ffmpeg`` as a format-conversion tool. This is not a native Python
package, so we can't make it trivially use with ``pip install``. Instead, you
need to install the appropriate package for your operating system:

- macOS / OS X. Make sure you have Homebrew installed (see our :ref:`installation
  instructions <get_python3>` for some guidance). Then you can install it with:

  .. code-block:: bash

  	$ brew install ffmpeg

- Ubuntu 14.04 (Trusty Tahr). `Ubuntu switched
  <http://askubuntu.com/a/432585>`_ to ``avconv`` briefly as a replacement for
  ``ffmpeg``. So you can install a similar tool with:

  .. code-block:: bash

  	$ sudo apt-get install libav-tools

- Ubuntu 16.04 (Xenial Xerus). ``ffmpeg`` made a return in Ubuntu 15.04, so you
  can install it with:

  .. code-block:: bash

  	$ sudo apt-get install ffmpeg

- Arch Linux. Pretty simple:

  .. code-block:: bash

  	$ sudo pacman -S ffmpeg

Plotting
========

.. _fix_matplotlib:

I get an error from `matplotlib` saying "Python is not installed as a framework."
---------------------------------------------------------------------------------

Did you use ``pip`` to install ``matplotlib`` (you should!). The trick is to
tell ``matplotlib`` which "backend" it should use for its plotting. Do this:

.. code-block:: bash

	echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc

There, that should do it.

Plots from ``matplotlib`` don't appear, or I get errors about backends.
-----------------------------------------------------------------------

See :ref:`this answer <fix_matplotlib>`.

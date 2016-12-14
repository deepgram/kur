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
our :doc:`installing` guide), then the virtual environment will automatically
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

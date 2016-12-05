**************
Installing Kur
**************

Ready to install Kur? Let's get going!

.. note::

	If you want to install Kur for the purpose of developing, modifying, or
	contributing to Kur, then take a look at :doc:`contributing`.

Setting Up an Environment
=========================

Getting Python 3
----------------

Kur requires **Python 3**. Don't know what version you have? Pull up a terminal
and check!

.. code-block:: bash

	$ python --version

If your Python version is 2, let's just double-check that you don't have Python 3 installed as a different executable:

.. code-block:: bash

	$ python3 --version

If either of those command works and you have Python 3.4 or greater installed, you are all set! If you have Python 3, but it is older than 3.4, you need to upgrade (using whatever method you used to install Python 3 originally). If you only have Python 2, it's time to move into the future--let's install it!

Virtual Environments
--------------------

This step is optional, but highly recommended, since virtual environments allow
you to isolate different packages and package versions, making installations
cleaner, more reliable, and more stable.

Installing Kur
==============

You can either install the latest official release from PyPI, or the bleeding-edge development version from GitHub. You only need to pick one.

.. note::

	If you are using a virtual environment, make sure it is activated before
	continuing.

From PyPI
---------

Wow. This is easy:

.. code-block:: bash

	$ pip install kur

From GitHub
-----------

This is really easy, too. Just clone the repository and install:

.. code-block:: bash

	$ git clone https://github.com/deepgram/kur
	$ cd kur
	$ pip install .

.. note::

	If you run the install script ``python setup.py install``, then Python will
	try to build dependencies (like Numpy) from source. If you don't have the
	appropriate development environment (C compiler, FORTRAN compiler, etc.),
	then this will fail. It's much easier to just use ``pip`` for the
	installation.

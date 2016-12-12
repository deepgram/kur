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

	python --version

If your Python version is 2, let's just double-check that you don't have Python
3 installed as a different executable:

.. code-block:: bash

	python3 --version

If either of those command works and you have Python 3.4 or greater installed,
you are all set! If you have Python 3, but it is older than 3.4, you need to
upgrade (using whatever method you used to install Python 3 originally). If you
only have Python 2, it's time to move into the future--let's install it!

- OS X. You have a few options for installing Python 3 on a Mac. If you've
  never installed Python 3 before, we recommend doing the following:

	#. Install a C compiler. The easiest way to do this is to install XCode
	   from the Apple Store (it's free). Then open up a terminal (the Terminal
	   utility is in the Applications | Utilities folder) and type:

		.. code-block:: bash

			xcode-select --install

		Proceed through any windows or confirmations that come up.

	#. Install `Homebrew <http://brew.sh>`_. To do this, type the following in
	   a terminal:

		.. code-block:: bash

			/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

	#. Let's make sure that everything you install using Homebrew is "on your
	   path" (so your system knows where Python 3 and other Homebrew programs
	   live). Open a terminal, and type:

	   	.. code-block:: bash

			echo 'PATH=/usr/local/bin:$PATH' >> ~/.profile

	   Now, close the terminal---you need a fresh terminal for that change to
	   take effect.

	#. Actually install Python 3! Enter this into a terminal:

		.. code-block:: bash

			brew install python3

	#. Make sure everything worked. To do this, you need to open a fresh
	   terminal (it has to be a new terminal---you can't reuse the same
	   terminal you just used in the previous steps). Then do this:

	   	.. code-block:: bash

			python3 --version

	   If everything worked, you should see the version of Python 3 you just
	   installed appear on the screen. Great!

- Linux. Installing Python 3 depends on your Linux distribution; most new Linux
  releases are including Python 3 installed as the default Python interpreter.
  But obviously you got this far into the installation instructions, so that
  isn't the case for your current distribution!

  For Ubuntu, you can do this:

  	.. code-block:: bash

		sudo apt-get update
		sudo apt-get install python3 python3-pip

  For other distributions, please refer to your distribution's package manager
  and repositories to determine the exact name of the Python 3 package (and how
  to install it). Make sure you install ``pip`` for Python 3, too.

.. _virtualenv_setup:

Virtual Environments
--------------------

This step is optional, but **highly** recommended, since virtual environments
allow you to isolate different packages and package versions, making
installations cleaner, more reliable, and more stable.

Let's install the core package and its highly convenient helper utility:

.. code-block:: bash

	pip install virtualenv virtualenvwrapper

We also need to update your profile. Following these instructions, depending on
your platform.

	- OS X:

	  .. code-block:: bash

		echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.profile
		echo 'source $(which virtualenvwrapper.sh)' >> ~/.profile
		source ~/.profile

	- Linux: this depends on your shell. For ``bash`` (which is very common for
	  Linux distributions to use), do this:

	  .. code-block:: bash

		echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.bashrc
		echo 'source $(which virtualenvwrapper.sh)' >> ~/.bashrc
		source ~/.bashrc

.. note::

	Different systems install ``virtualenvwrapper.sh`` in different locations.
	Lots of them do something intelligent, so that the above instructions for
	updating your profile work. However, if you start seeing errors from your
	shell that look like this::

		-bash: source: filename argument required
		source: usage: source filename [arguments]

	or this::

		source: no such file or directory: virtualenvwrapper.sh

	then you know that your system has put the script in a silly place. First,
	we need to find out where it is::

		find / -name virtualenvwrapper.sh 2>/dev/null

	Then edit your profile (using ``vim``, ``emacs``, ``nano``, etc.) and
	change this line::

		source $(which virtualenvwrapper.sh)

	to this::

		source /path/to/virtualenvwrapper.sh
	
	replacing ``/path/to/virtualenvwrapper.sh`` with the path outputted by the
	``find`` command.

Now you should create a virtual environment for Kur:

.. code-block:: bash

	mkvirtualenv -p /usr/bin/python3 kur

This will create and "activate" the Kur virtual environment. You can
"deactivate" the virtual environment with this command:

.. code-block:: bash

	deactivate

To activate the virtual environment (which you should do anytime you want to
use Kur), do this:

.. code-block:: bash

	workon kur

Installing Kur
==============

Setting Up a Virtual Environment
--------------------------------

First things first: make sure your virtual environment is set up, so that Kur
and its dependencies can reside in a happy, isolated environment from your
other Python packages. *If you really don't want to do this, just continue on.*
But you really should take a moment and follow along with
:ref:`virtualenv_setup`.

Now all you have to do is make sure your environment is activated:

.. code-block:: bash

	workon kur

Getting the Package
-------------------

You can either install the latest official release from PyPI, or the
bleeding-edge development version from GitHub. You only need to pick one.

From PyPI
^^^^^^^^^

Wow. This is easy:

.. code-block:: bash

	pip install kur

From GitHub
^^^^^^^^^^^

This is really easy, too. Just clone the repository and install:

.. code-block:: bash

	git clone https://github.com/deepgram/kur
	cd kur
	pip install .

.. note::

	If you run the install script ``python setup.py install``, then Python will
	try to build dependencies (like Numpy) from source. If you don't have the
	appropriate development environment (C compiler, FORTRAN compiler, etc.),
	then this will fail. It's much easier to just use ``pip`` for the
	installation.

	Also, if you are interested in contributing to or modifying Kur, then you
	probably want to install the package using ``pip install -e .``. See
	:doc:`contributing` for details.

Verifying the Installation
--------------------------

If everything has gone well, you shoud be able to use Kur:

.. code-block:: bash

	kur --version

If Kur prints out a version, everything is working great! Now move on to the
:ref:`the_examples` or the :doc:`tutorial` and start building awesome models!

Usage
-----

You can look at Kur's usage like this:

.. code-block:: bash

	kur --help

You'll typically be using Kur in commands like ``kur train model.yml`` or ``kur
test model.yml``. You'll see these in the :ref:`the_examples`, which is where
you should head to next!

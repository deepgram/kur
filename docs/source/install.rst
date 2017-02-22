.. _install_kur:

**************
Installing Kur
**************

Kur Quick Install
=================

Kur is really easy to install! You can pick either one of these two options for
installing Kur.

.. note::

	Kur requires **Python 3.4** or greater. Take a look at :ref:`detailed_install`
	for step-by-step instructions for installing Kur and setting up a `virtual
	environment <https://virtualenv.pypa.io/>`_.

Latest Pip Release
------------------

If you know what you are doing, then this is easy:

.. code-block:: bash

	pip install kur

Latest Development Release
--------------------------

Just check it out and run the setup script:

.. code-block:: bash

	git clone https://github.com/deepgram/kur
	cd kur
	pip install .

**Quick Start**: Or, if you already have :ref:`Python 3 installed
<detailed_install>`, then here's a few quick-start lines to get you training your
first model:

**Quick Start For Using pip:**

.. code-block:: bash

	pip install virtualenv                      # Make sure virtualenv is present
	virtualenv -p $(which python3) ~/kur-env    # Create a Python 3 environment for Kur
	. ~/kur-env/bin/activate                    # Activate the Kur environment
	pip install kur                             # Install Kur
	kur --version                               # Check that everything works
	git clone https://github.com/deepgram/kur   # Get the examples
	cd kur/examples                             # Change directories
	kur -v train mnist.yml                      # Start training!

**Quick Start For Using git:**

.. code-block:: bash

	pip install virtualenv                      # Make sure virtualenv is present
	virtualenv -p $(which python3) ~/kur-env    # Create a Python 3 environment for Kur
	. ~/kur-env/bin/activate                    # Activate the Kur environment
	git clone https://github.com/deepgram/kur   # Check out the latest code
	cd kur                                      # Change directories
	pip install .                               # Install Kur
	kur --version                               # Check that everything works
	cd examples                                 # Change directories
	kur -v train mnist.yml                      # Start training!

Usage: Kur
----------

If everything has gone well, you shoud be able to use Kur:

.. code-block:: bash

	kur --version

You'll typically be using Kur in commands like ``kur train model.yml`` or ``kur
test model.yml``. You'll see these in the :ref:`in_depth_examples`, which is
where you should head to next!

Troubleshooting
---------------

If you run into any problems installing or using Kur, please check out our
:doc:`troubleshooting` page for lots of useful help. And if you want more
detailed installation instructions, with help on setting up your environment,
before sure to follow along in the next :ref:`detailed_install` section.

.. _detailed_install:

Detailed Kur Install Guide
==========================

Ready to install Kur but need a little more help than the Quick Install provides? This is the place!

.. note::

	If you want to install Kur for the purpose of developing, modifying, or
	contributing to Kur, then take a look at :doc:`contributing`.

How The Following Guide Helps
--------------------------------
This detailed installation guive can tell you in detail how to set up your environment to have a long lasting and organized experience while deep learning with Kur. There are many helpful suggestions for both Linux and Mac OSX users.

These docs won't be able to help with all possible problems that can arise while setting up a development environment. We strive to make these documents helpful to 95% of people, but it cannot cover all flavors of Linux and complicated constraints on your computers environment. 

With that said, these docs should get the vast majority of normal users up and running with Kur and Deep Learning in no time. Try to be patient during this process. Grab a cup of coffee and really think out how you want things set up on your computer. You'll be using Kur for years to come, so (it's that good). One last thing. Be sure to inform us with GitHub issues if you notice anything off in the docs and feel free to improve them—Kur is open source afterall!

Setting Up an Environment
=========================

.. _get_python3:

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
	   installed appear on the screen. Great! And what's more, Homebrew also
	   just installed ``pip3`` for you---it's a package manager for Python. To
	   make sure you have it, type ``which pip3``. Make sure to invoke ``pip3``
	   since ``pip`` may reference a different Python installation.

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
:ref:`in_depth_examples` or the :doc:`tutorial` and start building awesome models!

Usage
-----

You can view Kur's usage like this:

.. code-block:: bash

	kur --help

You'll typically be using Kur in commands like ``kur train model.yml`` or ``kur
test model.yml``. You'll see these in the :ref:`in_depth_examples`, which is where
you should head to next!

*******************
Contributing to Kur
*******************

Development Setup
=================

#. Check out the code:

	.. code-block:: bash

		git clone https://github.com/deepgram/kur
		cd kur

#. (Optional, but recommended) Set up a virtualenv.

	There are lots of ways to do this. The easiest way if you don't know how is
	to use `virtualenv <https://virtualenv.pypa.io/en/stable/>`_. You do this
	once:

	.. code-block:: bash

		virtualenv -p /usr/bin/python3.5 venv

	This will create a new folder in the repo root called ``venv`` (it is in
	the ``.gitignore``, so don't worry about it polluting anything).

	Now every time you are ready to work on Kur, activate the environment:

	.. code-block:: bash
	
		source venv/bin/activate

	This puts you in an isolated Python environment, with its own packages. If
	you install packages while the virtual environment is activatd, they will
	only be installed within the virtual environment, and the system packages
	will be left untouched.

	To leave the virtual environment, deactivate it:

	.. code-block:: bash

		deactivate

#. Install an editable version of Kur.

	.. code-block:: bash

		pip install -e .

	This will install Kur (within the virtual environment only, if one is
	active), but any changes you make to the Kur source code will be
	immediately "seen" by programs that use Kur (rather than having to
	remove/reinstall).

	.. note::

		This is very similar to the functionality provided by ``python setup.py
		develop``, but the unit testing framework that Kur uses (``pytest``) is
		slightly more annoying to run, as it won't "see" the main Kur package
		installed. If you really insist on using ``python setup.py develop``
		instead, then instead of running ``py.test``, you need to run
		``PYTHONPATH=.:$PYTHONPATH py.test`` or ``python -m pytest tests/``
		instead.

#. Install the unit-testing packages and ``pylint``.

	.. code-block:: bash

		pip install tox pytest pytest-xdist pylint

Running the Unit Tests
======================

Kur uses `pytest <http://pytest.org/>`_ as its unit-testing framework, and `tox
<https://tox.readthedocs.io/>`_ for running the unit tests in a number of
different, isolated environments (i.e., against different versions of Python,
each in their own virutal environment).

Running the Unit Tests with ``tox``
-----------------------------------

To run the entire unit-testing suite for all versions of Python, you can simply
do this:

.. code-block:: bash

	tox

.. note::

	Kur does not need to be installed to run the unit tests through ``tox``.
	This means that if you installed Kur in a virtual environment, you do not
	need to activate the virtual environment before running the unit tests
	(although there is no harm in running ``tox`` from within the virtual
	environment, too).

To run the unit-test suite through ``tox`` for a particular Python version (for
example, Python 3.5):

.. code-block:: bash

	tox -e py35

You can enumerate all defined ``tox`` environments using ``tox -l``.

Running the Unit Tests with ``pytest``
--------------------------------------

``tox`` already uses ``pytest`` behind the scenes to run the unit tests. But if
you want to run the tests manually, you can do so. They will only test against
the current Python environment.

.. code-block:: bash

	python -m pytest --boxed tests/

.. note::

	Unlike running the unit tests through ``tox``, if you want to call
	``pytest`` directly like this, you will need Kur installed (or your virtual
	environment activated).

.. note::

	Like we mentioned earlier, ``pytest`` is a little naïve about its Python
	path. If you installed Kur into a virtual environment, you'll need to tell
	``pytest`` where it is (even if the environment is already activated). If
	your virtual environment is called ``venv`` in the repository root, you can
	do (be sure to change your Python version as appropriate):

		.. code-block:: bash

			PYTHONPATH=venv/lib/python3.5/site-packages:$PYTHONPATH pytest

.. note::

	What's up with the ``--boxed`` option? It's an option for the
	``pytest-xdist`` plugin which runs each test in its own subprocess. This is
	important when testing backends like Keras, which do not seem to allow easy
	swapping of Theano/Tensorflow backends on-the-fly. Thus, when a test does
	``import keras``, the Keras backend will get "stuck" for that process.

Style Guide
===========

We loosely adhere to the :pep:`8` style guide. The most notable exception is
that our code is indented with **tabs** instead of **spaces**. Why? Although
Python suggests using spaces for indentation, spaces can be awkward to use:
they do not convey semantic information and they make it difficult for people
to adjust the indentation appearance to fit their preferences (on the other
hand, editors can usually be customized to display the tab character as any
number of spaces). Maybe this will change someday with enough public outcry.
For now, tabs rule.

We have a `Pylint <http://www.pylint.org/>`_ configuration file so that you,
too, can use the linter to check code quality. To do this, make sure ``pylint``
is installed (if it is in a virtual environment, make sure the environment is
activated) and then:

.. code-block:: bash

	pylint kur

Please make sure all linting issues are addressed before submitting a pull
request.

We do not lint our ``tests`` directory, because they break lots of rules due to
the magic of ``pytest`` (e.g., through fixtures and ``conftest.py`` files).

Bug Reporting
=============

Bugs should be reported as issues on GitHub. Please provide this information to
help us get things fixed!

- If you encountered a bug using the Python API:

	- Please actually think about the problem yourself a little, and tell us
	  what you've tried to do to avoid the problem.
	- Please describe what you expected the code to do.
	- Please provide a minimal working example (the smallest program that
	  reproduces your error) in Python.
	- Please include debug-level output: ``kur -vv ...``

- If you encountered a bug using the specification file and command-line API:

	- Please provide a minimum working example in YAML.
	- Please provide the command-line invocation(s) used.
	- Please tell us what you expected to happen.
	- Please include debug-level output: ``kur -vv ...``

In both cases, if you bug needs a data source to reproduce, you should:

	- Check if the example data suppliers can be used to recreate your problem.
	  This is definitely the most convenient way to check your problem, since
	  we don't need to download and understand your data.
	- If the examples don't cut it, see if you can include an example Numpy
	  array that produces the problem, either hard-coded or via some little
	  Python snippet that creates the array (e.g., with ``numpy.random``).
	- As a very last resort, you can try submitting **small** datasets (with as
	  few elements in them as possible to reproduce the problem). But doing
	  this will very likely deter us from addressing your issue, because it is
	  more frustrating having to deal with dataset problems than actual Kur
	  problems.

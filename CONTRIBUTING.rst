*******************
Contributing to Kur
*******************

Development Setup
=================

#. Check out the code:

	.. code-block:: bash

		$ git clone https://github.com/deepgram/kur
		$ cd kur

#. (Optional, but recommended) Set up a virtualenv.

	There are lots of ways to do this. The easiest way if you don't know how is
	to use `virtualenv <https://virtualenv.pypa.io/en/stable/>`_. You do this
	once:

	.. code-block:: bash

		$ virtualenv -p /usr/bin/python3.5 venv

	This will create a new folder in the repo root called ``venv`` (it is in the
	``.gitignore``, so don't worry about it polluting anything).

	Now every time you are ready to work on Kur, activate the environment:

	.. code-block:: bash
	
		$ source venv/bin/activate

	This puts you in an isolated Python environment, with its own packages. If
	you install packages while the virtual environment is activatd, they will
	only be installed within the virtual environment, and the system packages
	will be left untouched.

	To leave the virtual environment, deactivate it:

	.. code-block:: bash

		$ deactivate

#. Install an editable version of Kur.

	.. code-block:: bash

		$ pip install -e .

	This will install Kur (within the virtual environment only, if one is active), but any changes you make to the Kur source code will be immediately "seen" by programs that use Kur (rather than having to remove/reinstall).

	.. note::

		This is very similar to the functionality provided by ``python setup.py develop``, but the unit testing framework that Kur uses (``pytest``) is slightly more annoying to run, as it won't "see" the main Kur package installed. If you really insist on using ``python setup.py develop`` instead, then instead of running ``py.test``, you need to run ``PYTHONPATH=.:$PYTHONPATH py.test`` or ``python -m pytest tests/`` instead.

#. Install the unit-testing packages.

	.. code-block:: bash

		$ pip install tox pytest

Running the Unit Tests
======================

Kur uses `pytest <http://pytest.org/>`_ as its unit-testing framework, and `tox <https://tox.readthedocs.io/>`_ for running the unit tests in a number of different, isolated environments (i.e., against different versions of Python, each in their own virutal environment).

Running the Unit Tests with ``tox``
-----------------------------------

To run the entire unit-testing suite for all versions of Python, you can simply do this:

.. code-block:: bash

	$ tox

.. note::

	Kur does not need to be installed to run the unit tests through ``tox``.
	This means that if you installed Kur in a virtual environment, you do not
	need to activate the virtual environment before running the unit tests
	(although there is no harm in running ``tox`` from within the virtual
	environment, too).

To run the unit-test suite through ``tox`` for a particular Python version (for
example, Python 3.6):

.. code-block:: bash

	$ tox -e py46

You can enumerate all defined ``tox`` environments using ``tox -l``.

Running the Unit Tests with ``pytest``
--------------------------------------

``tox`` already uses ``pytest`` behind the scenes to run the unit tests. But if
you want to run the tests manually, you can do so. These are all equivalent, and
run all the unit tests.

.. code-block:: bash

	$ py.test
	$ pytest
	$ py.test tests/
	$ pytest tests/
	$ python -m pytest tests/
	$ pytest tests/path/to/test.py

.. note::

	Unlike running the unit tests through ``tox``, if you want to call
	``pytest`` directly like this, you will need Kur installed (or your virtual
	environment activated).

Style Guide
===========


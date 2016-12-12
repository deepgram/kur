Building Documentation
======================

Installing Sphinx
-----------------

Kur's documentation is written in restructured text and built with
`Sphinx <http://www.sphinx-doc.org>`_. Installing Sphinx is easy:

.. code:: bash

	pip install sphinx

Install the Theme
-----------------

To build the documentation, you need to install the
`Read the Docs Sphinx theme <https://github.com/snide/sphinx_rtd_theme>`_.
To do this:

.. code:: bash

	pip install sphinx_rtd_theme

Actually Building Everything
----------------------------

From the root of the repository, enter the documentation directory (``cd docs``)
and then you can build the documentation with:

.. code:: bash

	make html

This will put the root HTML page at ``docs/build/html/index.html`` relative to
the root of the repository.

.. note::

	You don't need to install Kur or its dependencies to build the
	documentation.

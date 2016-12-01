**************************
Frequently Asked Questions
**************************

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

Do you support Python 2?
------------------------

No, and we won't. Python 3 was released in 2008, and Python 2 was last released
in 2009. Python 3 is definitely the more modern, streamlined, convenient, and
(increasingly) supported language. Rather than make our code ugly with lots of
hacks (like using ``__future__``) or restricting ourselves to a common grammar
that both languages support, we are going to continue working in Python 3.

We will occassionally bend a little to include Python 3.4 support, but we don't
support earlier versions.

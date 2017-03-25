*****************
Upcoming Features
*****************

Documentation and Support Features
==================================

- "Python API" overview of all components of the system
- Improve CIFAR-10 example with a custom evaluation hook.

Model Features
==============

- More intuitive error messages when trying to assemble models.
- Add "flatten" layers automatically before Dense layers.
- Add "activation" layers automatically between layers.
- Multi-model architectures, as used for GANs.

Container Features
-------------------

- Additional activation functions

Templating Features
-------------------

- Easily include model pieces from external sources (e.g., KurHub).

Loss/Optimizer Features
-----------------------

- New optimizers
- New loss functions

Data Features
=============

- More expressive data pipelines for creating derivative training sets,
  pre-processing data, and doing just-in-time manipulation of data.
- Column mapping: assign a column of data from a file to a model container with
  a different name.

Log Features
============

- Add a textual logger class.

Usage Features
==============

- Add "accuracy" statistic to the output during training/evaluation.
- Allow command-line arguments to override the specification file.
- Better detecting and handling unused keywords (which may have been typos).
- Standalone "validate" command.

Internal Features
=================

- Review providers: should they always return dictionaries?

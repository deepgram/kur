*****************
Upcoming Features
*****************

Documentation and Support Features
==================================

- "Python API" overview of all components of the system
- Additional examples: speech recognition
- Improve CIFAR-10 example with a custom evaluation hook.

Model Features
==============

- Rewrite the model construction to allow layers to have better context for
  what layers come before/after them. This will also allow layers to do shape
  inference.
- Add "flatten" layers automatically before Dense layers.
- Add "activation" layers automatically between layers.
- New "output" layer that simply is used for tagging a layer.

Containers Features
-------------------

- New containers: RNNs (LSTMs and GRUs)

Templating Features
-------------------

- Easily include model pieces from external sources (e.g., KurHub).

Loss/Optimizer Features
-----------------------

- New loss functions and optimizers

Data Features
=============

- More expressive data pipelines for creating derivative training sets,
  pre-processing data, and doing just-in-time manipulation of data.
- Column mapping: assign a column of data from a file to a model container with
  a different name.
- Fix multi-name merging (same column specified multiple times)

Log Features
============

- Add a textual logger class.

Usage Features
==============

- Add "accuracy" statistic to the output during training/evaluation.
- Allow command-line arguments to override the specification file.
- Better detecting and handling unused keywords (which may have been typos).
- Standalone "validate" command.
- The "build" command should optionally use/not use the data set.

Internal Features
=================

- Review providers: should they always return dictionaries?

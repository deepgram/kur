***************
Troubleshooting
***************

Installation
============

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

See `this answer <looks_stuck>`_.

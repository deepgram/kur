CHANGES
=======

master (unreleased)
-------------------

0.2.0 (2017.01.18)

- Massive overhaul of the Keras backend, moving away from the model extension
  pattern and toward low-level functions for implementing the training/testing/
  evaluation loops.
- Rewrote the model assembly code to be more flexible.
- Added new loss functions: CTC
- Added new optimizer: RMSProp
- Added new derived sources
- Added several new layers: recurrent, pooling, assertion, output, batch
  normalization, transpose.
- Better packaging for remote datasets.
- Added audio processing tools and a speech recognition example.
- By default, models block while compiling, so the user doesn't see a hanging
  "Epoch 1/X" prompt.
- Added support for shape tracking in the model.
- Updating documentation.
- Eliminated several dependencies.
- Improved unit tests and coverage.

0.1.2 (2016.12.05)
------------------

- Weights are now stored as directories rather than files. This structure is
  more flexible for transfer learning, and also frees Kur from the h5py
  dependency that was being carried from Keras.
- New auto-generated container names. Names now play nice with TensorFlow's
  naming requirements.
- The Keras backend now more carefully considers which backend to use.
- Improved unit tests for both the Theano and TensorFlow Keras backends.

0.1.1 (2016.12.01)
------------------

- Fixed the README file, which had an invalid reStructuredText tag from Sphinx,
  and which was messing up the PyPI page.

0.1.0 (2016.12.01)
------------------

- Initial release of Kur.

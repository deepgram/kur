CHANGES
=======

master (unreleased)
-------------------

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

CHANGES
=======

master (unreleased)
-------------------

0.5.2 (2017.03.28)

- Fixed a bug when processing meta-container inputs.
- Added globbing patterns for includes.

0.5.1 (2017.03.25)

- Fixed a bug in the Keras parallelizing which crashed TensorFlow while waiting
  for compilation to finish).
- Improved debug output for Keras 2.0.2.

0.5.0 (2017.03.24)

- Very small update prior to the Deep Learning Hackathon
- Simplifies the requirements for subclassing Container
- Added some documentation about the text hook.
- Added an initial KurHub hook to support the hackathon.

0.4.0 (2017.03.23)

- Improved GPU selection
- Documentation updates
- Better JSONL loading
- Couple minor bug fixes
- New layer: for_each
- Added templating and meta-containers

0.4.0rc0 (2017.03.15)

- New backend: PyTorch
- Multi-GPU support for PyTorch and TensorFlow.
- Compatibility updates to play nicely with Keras 2.0 and Theano 0.8/0.9.
- Character language model example.
- New layers (dropout, repeat,embedding)
- New sort options (i.e., neighborhood sort).
- Loop fill mode for the speech recognition supplier.
- Various bug and linting fixes.
- Code factoring in the Executor.
- Improved hooks (e.g., Slack hook with file uploads) and new hooks (plotting).
- Additional debug functionality (--step, --monitor, 'dump')
- Added checkpointing.
- Several fixes to the evaluation function.

0.3.0 (2017.01.25)

- New data supplier for CSV data!
- Hooks have been added to training, testing, and validation sections.
- Added a Slack training hook for keeping your team informed of training
  progress.
- Kur will transparently stack data sources when distributed across multiple
  files.
- Added a Merge layer.
- Integrated TensorFlow support for Python 3.6.
- "data" command added to Kur for checking data pipeline.
- Fixed problems with Kur selecting GPU devices.
- Fixed a localization problem with the PyPI description.
- More flexible vocabulary and normalization specification for ASR training.
- Minor typos corrected.

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

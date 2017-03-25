**************
Kur Python API
**************

In addition to the Kurfile approach to creating, training, and evaluating
models, Kur also exposes a Python API that can be used for advanced integration
with other systems.

.. note::

	This documentation page is under heavily development and updating. Lots of
	things are not covered here. In the meantime, feel free to ask us tons of
	questions on `Gitter <https://gitter.im/deepgram-kur/Lobby>`_!

Quick Start
===========

Here is a quick snippet to get started:

.. code-block:: python

	import numpy
	from kur import Kurfile
	from kur.engine import JinjaEngine

	# Load the Kurfile and parse it.
	kurfile = Kurfile('Kurfile.yml', JinjaEngine())
	kurfile.parse()

	# Get the model and assemble it.
	model = kurfile.get_model()
	model.backend.compile(model)

	# If we have existing weights, let's load them into the model.
	model.restore('weights.kur')

	# Now we can use the model for evaluating data.
	pdf, metrics = model.backend.evaluate(
	    model,
	    data={
	        'X' : numpy.array([1, 2, 3]),
	        'Y' : numpy.array([4, 5, 6]),
	    }
	)

Let's break it down.

#. **Load the Kurfile**. There's no reason to toss out the Kurfile! Otherwise,
   you're basically losing a lot of flexibility in expressing your awesome
   deep learning ideas. So let's load the file::

   	kurfile = Kurfile('Kurfile.yml', JinjaEngine())

   This will load the Kurfile up (it could be YAML or JSON), and tells Kur to
   use the Jinja2 templating engine to parse the variable references. Kur will
   go ahead and load the raw YAML/JSON (resolving YAML anchors on-the-fly), but
   will not do anything with the resulting data. So we have to actually tell
   Kur to parse the file::

   	kurfile.parse()

   Now we're in business!

#. **Prepare the model**. We already parsed the Kurfile, so now we need to pull
   out the model itself and evaluate it to determine the computation graph::

   	model = kurfile.get_model()

   And now we compile the resulting graph::

   	model.backend.compile(model)

   If we have previously trained weights, then we will want to load them up,
   too::

   	model.restore('weights.kur')

#. **Use the model**. Now we can actually use the model to evaluate new data.
   Data is fed in as a simply Python dictionary whose keys correspond to the
   layers referenced in the ``model`` (and the ``loss`` section, if you are
   training/testing). The values of this dictionary are just Numpy
   dictionaries. That's pretty simple! So how do we evaluate on a single batch?
   Easy:

   .. code-block:: python

	pdf, metrics = model.backend.evaluate(
		model,
		data={
			'X' : numpy.array([1, 2, 3]),
			'Y' : numpy.array([4, 5, 6]),
		}
	)

   Here, we're assuming you are doing _evaluation_ and that your model has two
   input layers named ``X`` and ``Y``. The numpy arrays can merrily be
   multi-dimensional, too. Almost always, each numpy array should be the same
   length. And since this evaluates on a single batch, you should keep batch
   sizes (i.e., the length of any given numpy array) reasonable or you'll run
   out of memory. If you have a ton of data to evaluate, just toss it into a
   loop!

Other Considerations
--------------------

Are all things this simple? Sometimes, yes; other times, no. So what sorts of
complications do you need to be aware of when using the Python API?

- Shapes. Normally, Kur does a really good job of inferring the correct shape
  of your data tensors. It can do this by looking at the data that the data
  suppliers load. However, in this example, we didn't use a data supplier (but
  you could have, if you wanted to!), and therefore Kur's shape inferencing may
  fail, giving an error like, "Placeholder requires a shape." In this case, you
  should explicitly specify the shapes of your input layers in the Kurfile.
  For example, replace an input layer that looks like this:

  .. code-block:: yaml

  	- input: X

  with one that looks like this:

  .. code-block:: yaml

  	- input:
	    shape: SHAPE
	  name: X

  ``SHAPE`` should be a list of input tensor shapes (excluding the batch size).
  If one of the dimensions is variable-length (like audio utterances in the
  speech recognition example), set it to ``null``. See the
  :ref:`ref_placeholder` documentation for more information.

- Data Preprocessing. Kur's sources and suppliers can take care of some data
  preprocessing for you. However, if you are going bare-bones and not using any
  of Kur's suppliers (as in this simple example), then you'll need to make sure
  you take care of these things. Specifically, think about these things:

  - Normalization. If you are using real-valued vectors, then your data
    should almost always be mean-subtracted and scaled before you submit it
    to Kur for training/evaluation. And of course, the parameters of this
    normalization (e.g., mean, variance) should be calculated at training
    time and then applied at testing/inference time.

  - One-hot. If you are trying to do a 1-of-N classification task (e.g.,
    image classification) or a M-of-N classification task (e.g., binary
    regression), then you probably want your outputs to be encoded as one-hot
    vectors. Similarly, some NLP tasks use one-hot representations. If your
    data is not already in a one-hot format, make sure you cast it correctly
    before sending it to Kur.

- Special data sources. Some of Kur's suppliers are really clever, and they
  generate additional data for you, as might be needed for sorting or for loss
  calculations. If you get errors about missing data sources, then you need to
  add in these additional data sources yourself. You can do this in two ways.
  First, you can simply use one of Kur's off-the-shelf Suppliers to help you
  with this task. Second, you can generate the data yourself and supply it to
  Kur at evaluation/training time. Pro-tip: use Kur's ``kur -vv train --step``
  and ``kur -vv data`` features to help you figure out what sorts of data are
  being sent into your model.

Can we simplify things so that we don't need to do all of this ourselves?
Absolutely! Let's just pull in some of Kur's other features, like using
BatchProvider and the existing data suppliers.

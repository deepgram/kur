********
Tutorial
********

Alright, you've seen some cool :doc:`examples` and now you are asking, "Okay, so
how do I actually make Kur do all these awesome things with my data?" Let's take
a look!

We are going to work through a complete, top-to-bottom model for classifying
2-dimensional points as being above or below a sine curve. I'm going to use
Python with Numpy for some non-Kur code (like generating data) because Python
is awesome. Okay, here's what we need to do:

- Generate our data
- Describe our model
- Run Kur
- Process the outputs

Let's go!

Generate Data
=============

Let's generate 2-dimensional points, where *x* is in [0, 2 pi] and *y* is in
[-1, 1]. We'll write a tiny Python script which takes two arguments: the number
of points to generate, and the file to save them to. Here's the script:

.. code-block:: python
	:caption: make_points.py

	import sys
	import pickle
	import numpy

	if len(sys.argv) != 3:
		print(
			'Usage: {} NUM-SAMPLES OUTPUT-FILE'.format(sys.argv[0]),
			file=sys.stderr
		)
		sys.exit(1)

	_, num_samples, output_file = sys.argv
	num_samples = int(num_samples)

	x = numpy.array([
		numpy.random.uniform(0, 2*numpy.pi, num_samples),
		numpy.random.uniform(-1, 1, num_samples)
	]).T
	y = (numpy.sin(x[:,0]) < x[:,1]).astype(numpy.float32)

	with open(output_file, 'wb') as fh:
		fh.write(pickle.dumps({'point' : x, 'above' : y}))

Alright. Let's make some data!

.. code-block:: bash

	$ python make_points.py 10000 train.pkl
	$ python make_points.py 1000 validate.pkl
	$ python make_points.py 1000 test.pkl
	$ python make_points.py 1000 evaluate.pkl

Let's make sure everything looks good:

	>>> import pickle
	>>> import numpy
	>>> with open('train.pkl', 'rb') as fh:
	...     data = pickle.loads(fh.read())
	...
	>>> list(data.keys())
	['above', 'point']
	>>> data['point'][:10]
	array([[ 2.67717122, -0.09930344],
	       [ 2.33520158, -0.2912654 ],
	       [ 5.05547674, -0.27683066],
	       [ 4.12045284,  0.73941201],
	       [ 4.21970901, -0.79627968],
	       [ 3.83711182,  0.97778992],
	       [ 1.54636165, -0.45812717],
	       [ 4.66270391,  0.26527582],
	       [ 3.78532394,  0.80580602],
	       [ 0.71286553, -0.37779473]])
	>>> data['above'][:10]
	array([ 0.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  0.], dtype=float32)

Good! Everything looks nice.

Describe the Model
==================

So what kind of model should we build? It's a tutorial, so let's build a classic
multi-layer perceptron (MLP) with one hidden layer. This type of model has two
fully-connected layers (input-to-hidden and hidden-to-out), and we will put a
non-linearity after each transformation.

The Model Itself
----------------

Let's start with the ``model`` section of the specification. How big does the
hidden layer need to be? Let's pick something like 128. How big does the last
layer need to be? Just 1, because our output is just scalars.

Also, we need to make sure the names of our inputs and outputs in the model
match the names of the data dictionary. We called the inputs ``point`` and we
called the outputs ``above``.

Putting it all together, we realize that our model looks like this:

.. code-block:: yaml

	model:
	  - input: point
	  - dense: 128
	  - activation: tanh
	  - dense: 1
	  - activation: tanh
	    name: above

The Operational Sections
------------------------

Now let's look at the "operational" sections: train, validate, test, evaluate.
The data is all in the same Python pickle format, and for the most part, we can
keep all of the default options. Let's train for ten epochs and, just in case
we want to train multiple times, let's make sure we reload our best-performing
weights (we respect to the validation weights, of course). Our ``train`` section
has got to look like this:

.. code-block:: yaml

	train:
	  data:
	    - pickle: train.pkl
	  epochs: 10
	  weights: best.w

The ``validate`` section is similar: we want to make sure we save the validation
weights. So it looks like:

.. code-block:: yaml

	validate:
	  data:
	    - pickle: validate.pkl
	  weights: best.w

The ``test`` section is exactly the same, except for the data file, since we
are using the same best-validation weights:

.. code-block:: yaml

	test:
	  data:
	    - pickle: test.pkl
	  weights: best.w

The ``evaluate`` section will also be similar, except we'll want to save the
outputs somewhere.

.. code-block:: yaml

	evaluate:
	  data:
	    - pickle: evaluate.pkl
	  weights: best.w
	  destination: output.pkl

There! That was easy.

The Loss Function
-----------------

The only thing missing is the loss function. What do we want to minimize? Well,
we want the model's outputs the be as close as possible to the true above/below
data. And everything is just scalars. So a really simple loss function to
minimize is mean-squared error.

We also need to assign the loss function to a model output, so we need to make
sure we keep the output names consistent: remember, it's "above", just like we
used in the data files and in the model.

.. code-block:: yaml

	loss:
	  - target: above
	    name: mean_squared_error

Running Kur
===========

Alright, do you have your data? Your specification file (make sure it starts
with ``---`` because it is YAML)? Assuming your specification file is named
``tutorial.yml``, let's train Kur:

.. code-block:: bash

	$ kur train tutorial.yml

	Epoch 1/10, loss=0.140: 100%|██████████████████| 10000/10000 [00:01<00:00, 7910.65samples/s]
	Validating, loss=0.126: 100%|████████████████████| 1000/1000 [00:00<00:00, 8739.94samples/s]

	Epoch 2/10, loss=0.125: 100%|█████████████████| 10000/10000 [00:00<00:00, 77666.83samples/s]
	Validating, loss=0.117: 100%|███████████████████| 1000/1000 [00:00<00:00, 95568.36samples/s]

	Epoch 3/10, loss=0.114: 100%|█████████████████| 10000/10000 [00:00<00:00, 76668.09samples/s]
	Validating, loss=0.108: 100%|███████████████████| 1000/1000 [00:00<00:00, 98395.48samples/s]

	Epoch 4/10, loss=0.105: 100%|█████████████████| 10000/10000 [00:00<00:00, 77204.35samples/s]
	Validating, loss=0.101: 100%|███████████████████| 1000/1000 [00:00<00:00, 97696.45samples/s]

	Epoch 5/10, loss=0.101: 100%|█████████████████| 10000/10000 [00:00<00:00, 76888.32samples/s]
	Validating, loss=0.099: 100%|███████████████████| 1000/1000 [00:00<00:00, 96879.57samples/s]

	Epoch 6/10, loss=0.099: 100%|█████████████████| 10000/10000 [00:00<00:00, 76661.92samples/s]
	Validating, loss=0.098: 100%|███████████████████| 1000/1000 [00:00<00:00, 96576.19samples/s]

	Epoch 7/10, loss=0.097: 100%|█████████████████| 10000/10000 [00:00<00:00, 76376.44samples/s]
	Validating, loss=0.094: 100%|███████████████████| 1000/1000 [00:00<00:00, 99067.13samples/s]

	Epoch 8/10, loss=0.095: 100%|█████████████████| 10000/10000 [00:00<00:00, 76825.65samples/s]
	Validating, loss=0.098: 100%|███████████████████| 1000/1000 [00:00<00:00, 96591.76samples/s]

	Epoch 9/10, loss=0.091: 100%|█████████████████| 10000/10000 [00:00<00:00, 77277.89samples/s]
	Validating, loss=0.091: 100%|███████████████████| 1000/1000 [00:00<00:00, 95275.29samples/s]

	Epoch 10/10, loss=0.087: 100%|████████████████| 10000/10000 [00:00<00:00, 75895.36samples/s]
	Validating, loss=0.083: 100%|███████████████████| 1000/1000 [00:00<00:00, 92804.60samples/s]

Everything is training beautifully. We can clearly see that both the training
set and the validation set are being used. Let's verify that we get comparable
loss on our test set:

.. code-block:: bash

	$ kur test tutorial.yml
	Testing, loss=0.087: 100%|███████████████████████| 1000/1000 [00:00<00:00, 1887.78samples/s]

Finally, let's evaluate the model on our evaluation set:

.. code-block:: bash

	$ kur evaluate tutorial.yml
	Evaluating: 100%|█████████████████████████████████| 1000/1000 [00:01<00:00, 766.88samples/s]

We just generated ``output.pkl``. Now let's take a look at it.

Post-processing
===============

Because our ``evaluate.pkl`` dataset contains the truth information ("above"),
the output file will contain both the model output as well as a copy of the
truth information.

Let's load things up and take a look.

	>>> import pickle
	>>> import numpy
	>>> with open('output.pkl', 'rb') as fh:
	...     data = pickle.loads(fh.read())
	...
	>>> list(data.keys())
	['truth', 'result']

Here ``result`` is the model prediction, and ``truth`` is the ground truth
information copied over from ``evaluate.pkl``. If no truth information was
available in the data file, then the ``truth`` key simply wouldn't be present in
this output file.

	>>> list(data['truth'].keys())
	['above']
	>>> list(data['result'].keys())
	['above']
	>>> type(data['truth']['above'])
	<class 'numpy.ndarray'>
	>>> type(data['result']['above'])
	<class 'numpy.ndarray'>
	>>> data['truth']['above'][:5]
	array([[ 1.],
		   [ 1.],
		   [ 1.],
		   [ 1.],
		   [ 0.]], dtype=float32)
	>>> data['result']['above'][:5]
	array([[ 0.93434536],
		   [ 0.64532757],
		   [ 0.9678849 ],
		   [ 0.91226113],
		   [ 0.20279442]], dtype=float32)

So we see that in both cases, the name of the model output has been copied over,
and it contains the numpy array. So the structure of our output file is this
(in YAML):

.. code-block:: yaml

	truth:
		above: numpy.array(...)
	result:
		above: numpy.array(...)

Our model has been trained to produce outputs closer to 0 whenever the ground
truth was 0, and to produce outputs closer to 1 whenever the ground truth was
1.  So we can characterize the accuracy by asking if the model is closer to 1
than 0 when the ground truth is 1, and that the model is closer to 0 than 1
when the ground truth is 0.

	>>> diff = data['truth']['above'] - data['result']['above'] < 0.5
	>>> correct = diff.sum()
	>>> total = len(diff)

``diff`` is True if the output is closer to the right answer than the wrong
answer, and False otherwise. In Python, summing a boolean array is like counting
the number of Trues (because each True counts for 1, and each False counts for 0).
So let's see what our accuracy is:

	>>> correct / total * 100
	98.700000000000003

98.7% accuracy! Pretty awesome!

.. note::

	The post-processing steps can be tedious at times. Kur supports the concept
	of a "hook" as means of extending Kur to do this analysis for you. If you
	have some programming skills and want to write custom hooks, you'll probably
	be glad you did!


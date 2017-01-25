.. |LICENSE| image:: https://img.shields.io/badge/license-Apache%202-blue.svg
   :target: https://github.com/deepgram/kur/blob/master/LICENSE
.. |PYTHON| image:: https://img.shields.io/badge/python-3.4%2C3.5%2C3.6-lightgrey.svg
   :target: https://kur.deepgram.com/installing.html
.. |BUILD| image:: https://travis-ci.org/deepgram/kur.svg?branch=master
   :target: https://travis-ci.org/deepgram/kur
.. |GITTER| image:: https://badges.gitter.im/deepgram-kur/Lobby.svg
   :target: https://gitter.im/deepgram-kur/Lobby

.. _Facebook: https://www.facebook.com/sharer/sharer.php?u=https%3A//kur.deepgram.com
.. _Google+: https://plus.google.com/share?url=https%3A//kur.deepgram.com
.. _LinkedIn: https://www.linkedin.com/shareArticle?mini=true&url=https%3A//kur.deepgram.com&title=Kur%20-%20descriptive%20deep%20learning&summary=Kur%20is%20the%20future%20of%20deep%20learning%3A%20advanced%20AI%20without%20programming!&source=
.. _Twitter: https://twitter.com/home?status=%40DeepgramAI%20has%20released%20the%20future%20of%20deep%20learning.%20https%3A//kur.deepgram.com%20%23Kur

.. image:: http://kur.deepgram.com/images/logo-small.png
   :align: center
   :target: https://deepgram.com

.. package_readme_starts_here

.. _Tutorial: https://kur.deepgram.com/tutorial.html

******************************
Kur: Descriptive Deep Learning
******************************

.. package_readme_ignore

|BUILD| |LICENSE| |PYTHON| |GITTER|

Introduction
============

Welcome to Kur! You've found the future of deep learning!

- Install Kur easily with ``pip install kur``.
- Design, train, and evaluate models *without ever needing to code*.
- Describe your model with easily understandable concepts, rather than trudge
  through programming.
- Quickly explore better versions of your model with the power of the `Jinja2
  <http://jinja.pocoo.org>`_ templating engine.
- **COMING SOON**: Share your models with the community, making it incredibly
  easy to collaborate on sophisticated models.

Go ahead and give it a whirl: `Get the Code`_ and then jump into
the `Examples`_! Then build your own model in our Tutorial_. Remember to check
out our `homepage <https://kur.deepgram.com>`_ for complete documentation and
the newest news.

.. package_readme_ignore

Like us? Share!

.. package_readme_ignore

- Facebook_
- `Google+`_
- LinkedIn_
- Twitter_

What is Kur?
------------

Kur is a system for quickly building and applying state-of-the-art deep
learning models to new and exciting problems. Kur was designed to appeal to the
entire machine learning community, from novices to veterans. It uses
specification files that are simple to read and author, meaning that you can
get started building sophisticated models *without ever needing to code*. Even
so, Kur exposes a friendly and extensible API to support advanced deep learning
architectures or workflows. Excited? Jump straight into the `Examples`_.

.. _get_the_code:

Get the Code
============

Kur is really easy to install! You can pick either one of these two options for
installing Kur.

**NOTE**: Kur requires **Python 3.4** or greater. Take a look at our
`installation guide <https://kur.deepgram.com/installing.html>`_ for
step-by-step instructions for installing Kur and setting up a `virtual
environment <https://virtualenv.pypa.io/>`_.

Latest Pip Release
------------------

If you know what you are doing, then this is easy:

.. code-block:: bash

	pip install kur

Latest Development Release
--------------------------

Just check it out and run the setup script:

.. code-block:: bash

	git clone https://github.com/deepgram/kur
	cd kur
	pip install .

**Quick Start**: Or, if you already have `Python 3 installed
<https://kur.deepgram.com/installing.html>`_, then here's a few quick-start
lines to get you training your first model:

**Quick Start For Using pip:**

.. code-block:: bash

	pip install virtualenv                      # Make sure virtualenv is present
	virtualenv -p $(which python3) ~/kur-env    # Create a Python 3 environment for Kur
	. ~/kur-env/bin/activate                    # Activate the Kur environment
	pip install kur                             # Install Kur
	kur --version                               # Check that everything works
	git clone https://github.com/deepgram/kur   # Get the examples
	cd kur/examples                             # Change directories
	kur train mnist.yml                         # Start training!

**Quick Start For Using git:**

.. code-block:: bash

	pip install virtualenv                      # Make sure virtualenv is present
	virtualenv -p $(which python3) ~/kur-env    # Create a Python 3 environment for Kur
	. ~/kur-env/bin/activate                    # Activate the Kur environment
	git clone https://github.com/deepgram/kur   # Check out the latest code
	cd kur                                      # Change directories
	pip install .                               # Install Kur
	kur --version                               # Check that everything works
	cd examples                                 # Change directories
	kur train mnist.yml                         # Start training!

Usage
-----

If everything has gone well, you shoud be able to use Kur:

.. code-block:: bash

	kur --version

You'll typically be using Kur in commands like ``kur train model.yml`` or ``kur
test model.yml``. You'll see these in the `Examples`_, which is
where you should head to next!

Troubleshooting
---------------

If you run into any problems installing or using Kur, please check out our
`troubleshooting <https://kur.deepgram.com/troubleshooting.html>`_ page for
lots of useful help. And if you want more detailed installation instructions,
with help on setting up your environment, before sure to see our `installation
<https://kur.deepgram.com/installing.html>`_ page.

.. package_readme_ends_here

.. _the_examples:

Examples
********

Let's look at some examples of how fun and easy Kur makes state-of-the-art deep
learning.

.. _mnist_example:

MNIST: Handwriting recognition
==============================

Let's jump right in and see how awesome Kur is! The first example we'll look at
is Yann LeCun's `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset. This is a
dataset of 28x28 pixel images of individual handwritten digits between 0 and 9.
The goal of our model will be to perform image recognition, tagging the image
with the most likely digit it represents.

**NOTE**: As with most command line examples, lines preceded by ``$`` are lines
that you are supposed to type (followed by the ``ENTER`` key). Lines without an
initial ``$`` are lines which are printed to the screen (you don't type them).

First, you need to `Get the Code`_! If you installed via
``pip``, you'll need to checkout the ``examples`` directory from the
repository, like this:

.. code-block:: bash

	git clone https://github.com/deepgram/kur
	cd kur/examples

If you installed via ``git``, then you alreay have the ``examples`` directory
locally, so just move into the example directory:

.. code-block:: bash

	$ cd examples

Now let's train the MNIST model. This will download the data directly from the
web, and then start training for 10 epochs.

.. code-block:: bash

	$ kur train mnist.yml
	Downloading: 100%|█████████████████████████████████| 9.91M/9.91M [03:44<00:00, 44.2Kbytes/s]
	Downloading: 100%|█████████████████████████████████| 28.9K/28.9K [00:00<00:00, 66.1Kbytes/s]
	Downloading: 100%|█████████████████████████████████| 1.65M/1.65M [00:31<00:00, 52.6Kbytes/s]
	Downloading: 100%|█████████████████████████████████| 4.54K/4.54K [00:00<00:00, 19.8Kbytes/s]

	Epoch 1/10, loss=1.524: 100%|███████████████████████| 480/480 [00:02<00:00, 254.97samples/s]
	Validating, loss=0.829: 100%|█████████████████████| 3200/3200 [00:03<00:00, 889.91samples/s]

	Epoch 2/10, loss=0.628: 100%|███████████████████████| 480/480 [00:02<00:00, 228.25samples/s]
	Validating, loss=0.533: 100%|████████████████████| 3200/3200 [00:03<00:00, 1046.12samples/s]

	Epoch 3/10, loss=0.547: 100%|███████████████████████| 480/480 [00:02<00:00, 185.77samples/s]
	Validating, loss=0.491: 100%|████████████████████| 3200/3200 [00:03<00:00, 1030.57samples/s]

	Epoch 4/10, loss=0.488: 100%|███████████████████████| 480/480 [00:02<00:00, 225.42samples/s]
	Validating, loss=0.443: 100%|████████████████████| 3200/3200 [00:03<00:00, 1046.23samples/s]

	Epoch 5/10, loss=0.464: 100%|███████████████████████| 480/480 [00:03<00:00, 115.17samples/s]
	Validating, loss=0.403: 100%|█████████████████████| 3200/3200 [00:04<00:00, 799.46samples/s]

	Epoch 6/10, loss=0.486: 100%|███████████████████████| 480/480 [00:03<00:00, 183.11samples/s]
	Validating, loss=0.400: 100%|████████████████████| 3200/3200 [00:02<00:00, 1134.17samples/s]

	Epoch 7/10, loss=0.369: 100%|███████████████████████| 480/480 [00:02<00:00, 214.10samples/s]
	Validating, loss=0.366: 100%|█████████████████████| 3200/3200 [00:04<00:00, 735.61samples/s]

	Epoch 8/10, loss=0.353: 100%|███████████████████████| 480/480 [00:03<00:00, 204.33samples/s]
	Validating, loss=0.351: 100%|████████████████████| 3200/3200 [00:02<00:00, 1147.05samples/s]

	Epoch 9/10, loss=0.399: 100%|███████████████████████| 480/480 [00:02<00:00, 219.17samples/s]
	Validating, loss=0.343: 100%|████████████████████| 3200/3200 [00:02<00:00, 1149.07samples/s]

	Epoch 10/10, loss=0.307: 100%|██████████████████████| 480/480 [00:02<00:00, 220.97samples/s]
	Validating, loss=0.324: 100%|████████████████████| 3200/3200 [00:02<00:00, 1142.78samples/s]

What just happened? Kur downloaded the MNIST dataset from LeCun's website, and
then trained a model for ten epochs. Awesome!

Now let's see how well our model actually performs:

.. code-block:: bash

	$ kur evaluate mnist.yml
	Evaluating: 100%|██████████████████████████████| 10000/10000 [00:06<00:00, 1537.74samples/s]
	LABEL     CORRECT   TOTAL     ACCURACY  
	0         969       980        98.9%
	1         1118      1135       98.5%
	2         910       1032       88.2%
	3         926       1010       91.7%
	4         923       982        94.0%
	5         735       892        82.4%
	6         871       958        90.9%
	7         884       1028       86.0%
	8         818       974        84.0%
	9         868       1009       86.0%
	ALL       9022      10000      90.2%

Wow! Across the board, we already have 90% accuracy for recognizing
handwritten digits, and we only used 0.8% of the training set! That's how
awesome Kur is.

Excited yet? Read on!

**NOTE**: Clever readers will notice that each training epoch only used 480
training samples. But MNIST provides 60,000 training samples total, so what
gives?  Simple: lots of us are running this code on consumer hardware; in fact,
I'm running this example on my tiny ultrabook on an Intel Core m7 CPU. As
you'll see in `Under the Hood`_, I truncate the training process to only train
on 10 batches of 32 samples each, just to make the training loop finish in a
reasonable amount of time. It's not cheating: you still get 90% accuracy! But
if you have awesome hardware, or just want to see how good your accuracy can
get, then by all means read on and we'll show you how to modify that.

Under the Hood
--------------

So what exactly is going on here? Let's take a look at the MNIST example
specification file:

.. code-block:: yaml

	train:
	  data:
	    - mnist:
	        images:
	          url: "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
	        labels:
	          url: "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

	model:
	  - input: images
	  - convolution:
	      kernels: 64
	      size: [2, 2]
	  - activation: relu
	  - flatten:
	  - dense: 10
	  - activation: softmax
	    name: labels

	include: mnist-defaults.yml

This is just plain, old `YAML <http://yaml.org>`_, a markup language meant to
be easy for humans to interpret (for a good overview of YAML language features,
look at the `Ansible overview
<https://docs.ansible.com/ansible/YAMLSyntax.html>`_).

There's a section to put the data. That's this:

.. code-block:: yaml

	train:
	  data:
	    - mnist:
	        images:
	          url: "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
	        labels:
	          url: "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

And then there's a spot to define your model:

.. code-block:: yaml

	model:
	  - input: images
	  - convolution:
	      kernels: 64
	      size: [2, 2]
	  - activation: relu
	  - flatten:
	  - dense: 10
	  - activation: softmax
	    name: labels

And there is an "include" part that just contains some default settings
(advanced users might want to tweak these---don't worry, it's still simple):

.. code-block:: yaml

	include: mnist-defaults.yml

Very simple! Kur downloaded our data directly from LeCun's website for us,
that's easy. But what goes into in a Kur model? Just a nice, gentle list of
things you want your deep learning model to do. Let's break it down:

- We have an ``input`` called ``images`` (yep, it's the same ``images`` from our
  ``train`` section).
- We pass the input to a ``convolution`` layer.
- We add a regularized linear unit ("ReLU") activation.
- We collapse (``flatten``) the high-dimensional output of a convolution into a
  nice, flat, 1-dimensional shape appropriate for sending into the
  fully-connected layers.
- We add a fully-connected (``dense``) layer with 10 outputs.
- We add a softmax activation (appropriate for classification tasks like MNIST),
  and mark it as producing labels (``name: labels``).

And that's it! It's pretty naïve: one convolution + activation +
fully-connected + activation.  But it works: we got 90% accuracy after only
showing it a small subset of the training set.

But let's think about make it more complicated. What if we want two
convolutional layers instead? Easy! Just add another ``convolution`` section to
the model.  We'll also add in another non-linearity (ReLU activation) between
the two convolutions.

.. code-block:: yaml

	model:
	  - input: images
	  - convolution:
	      kernels: 64
	      size: [2, 2]
	  - activation: relu
	  - convolution:
	      kernels: 64
	      size: [2, 2]
	  - activation: relu
	  - flatten:
	  - dense: 10
	  - activation: softmax
	    name: labels

We can also add more dense (fully-connected) layers. You probably want them
separated by activation layers, too. So if we add a 32-node fully-connected
layer to our model, it now looks like this:

.. code-block:: yaml

	model:
	  - input: images
	  - convolution:
	      kernels: 64
	      size: [2, 2]
	  - activation: relu
	  - convolution:
	      kernels: 64
	      size: [2, 2]
	  - activation: relu
	  - flatten:
	  - dense: 32
	  - activation: relu
	  - dense: 10
	  - activation: softmax
	    name: labels

Let's give it a try! Save your changes, a just run the same ``kur train
mnist.yml`` and ``kur evaluate mnist.yml`` commands from before.

**NOTE**: A more complex model will likely need more data. So be sure to look
at the tip in `More Advanced Things`_ to train on more of the data set.

If you want to know more, the YAML specification that Kur uses is described in
greater detail in our `Using Kur
<https://kur.deepgram.com/getting_started.html>`_ page.

.. _more_advanced_things:

More Advanced Things
--------------------

The one line in the ``mnist.yml`` specification that we didn't cover is the
``include: mnist-defaults.yml`` line. This is just a convenient way for us to
separate out the default behavior of the MNIST example.

If you tweak this file, probably the big thing you want to remove is the
``num_batches: 10`` line, which is what limits training to just the first 10
batches every epoch. Just delete the line or comment it out, and Kur will train
on the whole dataset.

A Better MNIST
--------------

90% is pretty good! But can we do better? Absolutely! Let's see how.

We need to build a more expressive, deeper model. We will use more
convolutional layers, with occassional pooling layers. 

.. code-block:: yaml

	model:
	  - input: images

	  - convolution:
	      kernels: 64
	      size: [2, 2]
	  - activation: relu

	  - convolution:
	      kernels: 96
	      size: [2, 2]
	  - activation: relu

	  - pool: [3, 3]

	  - convolution:
	      kernels: 96
	      size: [2, 2]
	  - activation: relu

	  - flatten:
	  - dense: [64, 10]

	  - activation: softmax
	    name: labels

So we have three convolutions with a 3-by-3 pooling layer in the middle, and
two fully-connected layers.  Try training this model: ``kur train mnist.yml``.
Then evaluate it to see how it does: ``kur eval mnist.yml``. We got better than
95% *by training on only 0.8% of the training set*.

What happens if we give it more data? Like we `mentioned above`__, we can
adjust the amount of data we give Kur by twiddling the ``num_batches`` entry in
the ``train`` section of ``mnist-defaults.yml``. Let's try using 5% of the
dataset.  To do this, we'll set ``num_batches: 94`` (because 5% of 60,000 is
3000, and for the default batch size of 32, this comes out to about 94
batches). Now try training and evaluating again. We got almost 98%!

__ more_advanced_things_

Don't stop now, let's train on the whole thing (just remove the ``num_batches``
line altogether, or set ``num_batches: null``). Still training only 10 epochs,
we got 98.6%. Wow. Let's compare this to state of the art, which Yann LeCun
tracks on the `MNIST website <http://yann.lecun.com/exdb/mnist/>`_. It looks
like the best error rate also uses convolutions and achieved a 0.23% error rate
(so 99.77% accuracy). With just a couple tweaks, we are already only a percent
away from the world's best. Kur rocks.

.. _cifar_10:

CIFAR-10: Image Classification
==============================

Okay, MNIST was pretty cool, but Kur can do much, much more. Imagine if you
wanted to have an arbitrary number of convolution layers. Imagine if each
convolution should have a different number of kernels. Imagine if you truly
want *flexibility*. You've come to the right place.

Flexibility: Variables
----------------------

Kur uses an *engine* to determine how do variable substitution. `Jinja2
<http://jinja.pocoo.org>`_ is the default templating engine, and it is very
powerful and extensible. Let's see how to use it!

Let's look at the `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
dataset. This is a image classification dataset of small 32 by 32 pixel color
(RGB) images, each with one of ten classes (airplane, automobile, bird, cat,
deer, dog, frog, horse, ship, truck). You might decide to start with a very
similar model to the MNIST example:

.. code-block:: yaml

	model:
	  - input: images
	  - convolution:
	      kernels: 64
	      size: [2, 2]
	  - activation: relu
	  - flatten:
	  - dense: 10
	  - activation: softmax
	    name: labels

We will start with a simple modification: let's make the convolution `size` a
variable, so we can easily change it later. We can do it like this:

.. code-block:: yaml

	settings:
	  cnn:
	    size: [2, 2]

	model:
	  - input: images
	  - convolution:
	      kernels: 64
	      size: "{{ cnn.size }}"
	  - activation: relu
	  - flatten:
	  - dense: 10
	  - activation: softmax
	    name: labels

Okay, what just happened? First, we added a ``settings:`` section. This section
is the appropriate place to declare variables, settings, and hyperparameters
that will be used by the model (or for training, evaluation, etc.). We declared
a variable named ``cnn`` with a nested ``size`` variable. In Python, this would
be equivalent to a dictionary: ``{"cnn": {"size": [2, 2]}}``.

Then we used the variable in the model's convolution layer: ``size: "{{
cnn.size }}"``.  This is standard Jinja2 grammar. The double-brackets indicate
that variable substitution should take place (without the brackets, we would
accidently assign ``size`` to the literal string "cnn.size", which doesn't make
sense). The variable we grab is ``cnn.size``, corresponding to the variables we
added in the ``settings`` section.

Cool! So we can use variables now. But how does that help us? It seems like we
just made it more complicated. Well, let's imagine if we added another
convolution layer. We already know how to add extra convolutions by just adding
another `convolution` block (and usually you want another `activation: relu`
layer, too). So this would look like:

.. code-block:: yaml

	settings:
	  cnn:
	    size: [2, 2]

	model:
	  - input: images
	  - convolution:
	      kernels: 64
	      size: "{{ cnn.size }}"
	  - activation: relu
	  - convolution:
	      kernels: 64
	      size: "{{ cnn.size }}"
	  - activation: relu
	  - flatten:
	  - dense: 10
	  - activation: softmax
	    name: labels

Ah! So now we can see why variablizing the convolution size was nice: if we
want to play with a model that uses different size kernels, we only need to
edit one line instead of two.

But there are still two problems we might encounter:

- What if we wanted to try out lots of models with different numbers of
  convolutions?
- What if we wanted to use *different* ``size`` or ``kernel`` values in each
  convolution?

Kur can do it!

Flexibility: Loops
------------------

Let's address the first problem: what if we want to make the number of
convolutions? Kur supports many "meta-layers" that it calls "operators." A
very simple operator is the classic `"for" loop
<https://en.wikipedia.org/wiki/For_loop>`_. This allows us to add many
convolution + activation layers at once. It looks like this:

.. code-block:: yaml

	settings:
	  cnn:
	    size: [2, 2]

	model:
	  - input: images
	  - for:
	      range: 2
	      iterate:
	        - convolution:
	            kernels: 64
	            size: "{{ cnn.size }}"
	        - activation: relu
	  - flatten:
	  - dense: 10
	  - activation: softmax
	    name: labels

This is equivalent to the version without the "for" loop. The ``for:`` loop
tells us to do everything in the ``iterate:`` section twice. (Why twice?
Because ``range: 2``.) And of course, we can variabilize the number of
iterations like this:

.. code-block:: yaml

	settings:
	  cnn:
	    size: [2, 2]
	    layers: 2

	model:
	  - input: images
	  - for:
	      range: "{{ cnn.layers }}"
	      iterate:
	        - convolution:
	            kernels: 64
	            size: "{{ cnn.size }}"
	        - activation: relu
	  - flatten:
	  - dense: 10
	  - activation: softmax
	    name: labels

Think about this for a minute. Does it make sense? It should. The model looks
like this:

- An ``input`` layer of images.
- A number of ``convolution`` and ``activation`` layers. How many?
  ``cnn.layers``, so 2.
- The rest of the model is as expected: a dense operation followed by an
  activation.

Flexibility: Variable-length Loops
----------------------------------

So we solved the problem of allowing for a variable number of convolutions. But
what if each convolution should use a different number of kernels (or sizes,
etc.)?  Well, Kur can happily handle this, too. In fact, the ``for:`` loop
already does most of the work. Every ``for:`` loop creates its own "local"
variable to let you know which iteration it is on. The default name for this
variable is ``index``. So if we want to use a different number of kernels for
each convolution, we can do this:

.. code-block:: yaml

	settings:
	  cnn:
	    size: [2, 2]
	    kernels: [64, 32]
	    layers: 2

	model:
	  - input: images
	  - for:
	      range: "{{ cnn.layers }}"
	      iterate:
	        - convolution:
	            kernels: "{{ cnn.kernels[index] }}"
	            size: "{{ cnn.size }}"
	        - activation: relu
	  - flatten:
	  - dense: 10
	  - activation: softmax
	    name: labels

Again, this is just Jinja2 substitution: we are asking for the ``index``-th
element of the ``cnn.kernels`` list. Each iteration of the ``for:`` loop
therefore grabs a different value for ``kernels:``. Cool, huh?

But we can do one better.

Flexibility: Filters
--------------------

The annoying thing about our current model is that nothing forces the ``layers``
value to be the same as the length of the ``kernels`` variable. If you make
really long (like, length seventeen) but leave ``layers`` at two, you probably
made a mistake. (Why did you put in seventeen layers but then only use the first
two in the loop?) What you really want is to make sure that ``layers`` is set to
the length of the ``kernels`` list. Or put another way, you want add as many
convolutions as you have kernels in the list.

Jinja2 supports a concept called "filters," which are basically functions that
you can apply to objects. You can even define your own filters. But what we
want right now is a way to get the length of a list. It's easy and it looks
like this:

.. code-block:: yaml

	settings:
	  cnn:
	    size: [2, 2]
	    kernels: [64, 32]

	model:
	  - input: images
	  - for:
	      range: "{{ cnn.kernels|length }}"
	      iterate:
	        - convolution:
	            kernels: "{{ cnn.kernels[index] }}"
	            size: "{{ cnn.size }}"
	        - activation: relu
	  - flatten:
	  - dense: 10
	  - activation: softmax
	    name: labels

You'll notice that the ``layers`` variable is gone, and we have this funky
``|length`` thing in the "for" loop's ``range``. This is standard Jinja2: the
``length`` filter returns the length of a list. So now we are asking the "for"
loop to iterate as many times as we have another kernel size.

This is really cool if you think about it. You want to add another convolution
to the network? *All you do is add it's size to the* ``kernels`` *list*. And
look!  You're model is now more general, more reuseable. You could have used
the same model for MNIST! Or CIFAR! Or many different applications.

This is the heart of the **Kur philosophy: you should describe your model once
and simply.** The specification *describes** your model: a bunch of
convolutions and then a fully-connected layer. You can specify the details (how
many convolutions, their parameters, etc.) elsewhere. The model should stay
elegant.

**NOTE**: Of course, it isn't always easy to write reusable models. And the
learning curve can get in the way. When we say that models should be "simple,"
we don't mean that you don't need to think about it. We mean that it should be
simple to use, simple to modify, and simple to share. A more general model is
elegant: making changes to it is easy (you only modify the settings). And this
makes it easier to reuse in new contexts or to share with the community.
Simplicity is power.

Actually Training a CIFAR-10 Model
----------------------------------

Great, we now have a simple, but powerful and general model. Let's train it. As
before, you'll need to ``cd examples`` first.

.. code-block:: bash

	kur train cifar.yml

Again, evaluation is just as simple:

.. code-block:: bash

	kur evaluate cifar.yml

Advanced Features
-----------------

The ``cifar.yml`` specification file is more complicated than the MNIST one,
mostly to expose you to some more knobs you can tweak. For example, you'll see
these lines in the ``train`` section:

.. code-block:: yaml

	provider:
	  batch_size: 32
	  num_batches: 2

As in the MNIST case, ``num_batches`` tells Kur to only train on that many
batches of data each epoch (mostly so that if you don't have a nice GPU, the
example still finishes in a reasonable amount of time). The ``batch_size`` value
indicates the number of training samples that should be used in each batch.

.. _using_binary_logger:

The ``train`` section also has a ``log: cifar-log`` line. This tells Kur to
save a log file to ``cifar-log`` (in the current working directory). This log
contains lots of interesting information about current training loss, batch
loss, and the number of epochs. By default, they are binary-encoded files, but
you can load them using the Kur API (in Python 3):

.. code-block:: python

	from kur.loggers import BinaryLogger
	data = BinaryLogger.load_column(LOG_PATH, STATISTIC)

where ``LOG_PATH`` is the path to the log file (e.g., ``cifar-log``) and
``STATISTIC`` is one of the logged statistics. ``data`` will be a `Numpy
<http://www.numpy.org/>`_ array. To find available statistics, just list the
available files in the ``LOG_PATH``, like this:

.. code-block:: bash

	$ ls cifar-log
	training_loss_labels
	training_loss_total
	validation_loss_labels
	validation_loss_total

For an example of using this log data, see our Tutorial_.

Another difference from the MNIST examples is that there are more files
referring to weights in the CIFAR specification. For example, in the
``validate`` section there is:

.. code-block:: yaml

	weights: cifar.best.valid.w

This tells Kur to save the best models weights (corresponding to the lowest
loss on the *validation* set) to ``cifar.best.valid.w``. Similarly, in the
``train`` section there is this:

.. code-block:: yaml

	weights:
	  initial: cifar.best.valid.w
	  save_best: cifar.best.train.w
	  last: cifar.last.w

The ``initial`` key tells Kur to try and load ``cifar.best.valid.w`` (the best
weights with respect to the *validation* loss) at the beginning of training. If
this file doesn't exist, nothing happens. This means that if you run the
training cycle many times (with many calls to ``kur train cifar.yml``), you
always "restart" from the best model weights.

We are also saving the best weights (with respect to the *training* loss) to
``cifar.best.train.w``.  The most recent weights are saved to ``cifar.last.w``. 

**NOTE**: The weights depend on the model architecture. Say you you train CIFAR
and produce ``cifar.best.valid.w``. Then you tweak the model in the
specification file. If you try to resume training (``kur train cifar.yml``),
Kur will try to load ``cifar.best.valid.w``. But the weights many not fit the
new architecture! So, to be safe, you should always delete (or backup) your
weight files before trying to train a fresh, tweaked model. In a production
environment, you probably want to have different sub-directories for each
variation/tweak to the model so that you never run into this problem.

The CIFAR-10 example also explicitly specifies an optimizer in the ``train``
section:

.. code-block:: yaml

	optimizer:
	  name: adam
	  learning_rate: 0.001

The optimizer function is set in the ``name`` field and all other parameters
(such as ``learning_rate``) are defined in the other fields. You can safely
change the optimizer without breaking backwards-compatibility with older weight
files.

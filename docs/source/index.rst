.. |LICENSE| image:: https://img.shields.io/badge/license-Apache%202-blue.svg
   :target: https://github.com/deepgram/kur/blob/master/LICENSE
.. |PYTHON| image:: https://img.shields.io/badge/python-3.4%2C3.5%2C3.6-lightgrey.svg
   :target: http://kur.deepgram.com/installing.html
.. |BUILD| image:: https://travis-ci.org/deepgram/kur.svg?branch=master
   :target: https://travis-ci.org/deepgram/kur
.. |GITTER| image:: https://badges.gitter.im/deepgram-kur/Lobby.svg
   :target: https://gitter.im/deepgram-kur/Lobby

.. _Facebook: https://www.facebook.com/sharer/sharer.php?u=https%3A//kur.deepgram.com
.. _Google+: https://plus.google.com/share?url=https%3A//kur.deepgram.com
.. _LinkedIn: https://www.linkedin.com/shareArticle?mini=true&url=https%3A//kur.deepgram.com&title=Kur%20-%20descriptive%20deep%20learning&summary=Kur%20is%20the%20future%20of%20deep%20learning%3A%20advanced%20AI%20without%20programming!&source=
.. _Twitter: https://twitter.com/home?status=%40DeepgramAI%20has%20released%20the%20future%20of%20deep%20learning.%20https%3A//kur.deepgram.com%20%23Kur
.. _GitHub: http://www.github.com/deepgram/kur
.. _KurHub: http://www.kurhub.com

.. image:: http://kur.deepgram.com/images/logo.png
   :width: 50%
   :align: center
   :target: https://deepgram.com

******************************
Kur: Descriptive Deep Learning
******************************

|BUILD| |LICENSE| |PYTHON| |GITTER|

Introduction
============

Welcome to Kur! You've found the future of deep learning!

- Install Kur easily with ``pip install kur``.
- Design, train, and evaluate models *without ever needing to code*.
- Describe your model with easily understandable concepts, rather than trudge
  through programming.
- Quickly explore better versions of your model with the power of `Jinja2
  <http://jinja.pocoo.org>`_ to automate Kurfile specification.
- Kur is open source and available at GitHub_
- **COMING SOON**: Share your models on KurHub_, making it incredibly
  easy to collaborate on models and learn from others.

Here's a good list to follow: 

- start :ref:`install_kur`
- jump into the :ref:`Quick Examples<quick_examples>`
- absorb the :ref:`In Depth Examples<in_depth_examples>`
- then build your own model in the :doc:`tutorial`

Like us? Share!

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
architectures or workflows. Excited? Jump straight into the
:ref:`in_depth_examples`.

How is Kur Different?
---------------------

Kur represents a new paradigm for thinking about, building, and using state of
the art deep learning models. Rather than thinking about your architecture as a
series of tensor operations (``tanh(W * x + b)``) and getting lost in all the
details, you can focus on **describing** the architecture you want to
instantiate. Kur does the rest.

The Kur philosophy is that you should describe your model once and simply.
Simple doesn't mean brainless, nor does it imply that you are limited in what
you can do. By "simple" we mean that models should be simple to use, simply to
modify, and simple to share. A flexible, more general model is elegant. And
this makes it easier to reuse in new contexts or to share with the community.
Kur's power lies in quickly making models that are both flexible and reusable.

Aside: Brief History Lesson
---------------------------

Decades ago, researchers wrote low-level code using highly optimized linear
algebra libraries and ran the code on CPUs. After the rise of General Purpose
Computing on GPUs (GPGPU), researchers modified their code to use CUDA or
OpenCL. Although this code was functionally identical, GPU computing
represented an incredible breakthrough in efficiency, as these new machine
learning models could train and predict in fractions of the time compared to
CPUs. Problematically, these programs were relatively hard-coded; exploring
different hyperparameters or architectures typically required detailed
knowledge of the code, and was fraught with ugly and bug-prone hacks.

Eventually, computer scientists began abstracting away the low-level, dirty
details of highly-optimized CUDA code, and projects like `Theano
<deeplearning.net/software/theano/>`_ and `TensorFlow
<https://www.tensorflow.org/>`_ were born. These tools are incredible in that
they hide many of the implementation details of working with hardware (i.e.,
CPUs and GPUs), and instead expose higher-level tensor operations to the
developer. Even then, the developer is forced to choose between building up
higher-level abstractions of deep learning primitives, or devolving to the
rigid or hacked models of earlier years.  Projects like `Keras
<https://keras.io/>`_ and `Lasagna <https://github.com/Lasagne/Lasagne>`_
emerged organically, driven by a need to more quickly and intuitively develop
deep learning models. Their primary genius is in providing an API that maps to
the way people actually think about the components of a deep learning network
(e.g., as a "LSTM layer" rather than as a series of tensor operations).

Kur is the natural progression of these tools and abstrations. It allows you,
the researcher, to get straight to the heart of deep learning: develop that
awesome model you've been dreaming about in a few short lines. And best of all,
you craft your model with high-level abstractions rather than having to think
about annoying questions like:

- Which language should I use?
- Which backend is the best?
- What if I want to quickly test different model configurations?

Indices and tables
==================

Contents:

.. toctree::
   :maxdepth: 1

   install
   in_depth_examples
   tutorial
   getting_started
   specification
   containers
   faq
   troubleshooting
   misc

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


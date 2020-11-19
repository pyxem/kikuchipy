============
Installation
============

kikuchipy can be installed from `Anaconda
<https://anaconda.org/conda-forge/kikuchipy>`_, the `Python Package Index
<https://pypi.org/project/kikuchipy/>`_ (``pip``), or from source, and only
supports Python >= 3.7.

We recommend you install it in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
with the `Miniconda distribution`_::

   $conda create --name kikuchipy python=3.8
   $conda activate kikuchipy

If you prefer a graphical interface to manage packages and environments, install
the `Anaconda distribution`_ instead.

.. _Miniconda distribution: https://docs.conda.io/en/latest/miniconda.html
.. _Anaconda distribution: https://docs.continuum.io/anaconda/

.. _install-with-anaconda:

Anaconda
--------

Anaconda provides the easiest installation. In the Anaconda Prompt, terminal or
Command Prompt, install with::

    $ conda install kikuchipy --channel conda-forge

If you at a later time need to update the package::

    $ conda update kikuchipy

.. _install-with-pip:

Pip
---

To install with ``pip``, run the following in the Anaconda Prompt, terminal or
Command Prompt::

    $ pip install kikuchipy

If you at a later time need to update the package::

    $ pip install --upgrade kikuchipy

.. note::

    kikuchipy builds upon HyperSpy, which depends upon a number of libraries
    that usually need to be compiled. Installing kikuchipy with ``pip`` may
    therefore require some development tools.

.. _install-from-source:

Install from source
-------------------

To install kikuchipy from source, clone the repository from `GitHub
<https://github.com/pyxem/kikuchipy>`_::

    $ git clone https://github.com/pyxem/kikuchipy.git
    $ cd kikuchipy
    $ pip install --editable .

See the :ref:`contributing guidelines <setting-up-a-development-installation>`
for how to set up a development installation.

============
Installation
============

kikuchipy can be installed from `Anaconda
<https://anaconda.org/conda-forge/kikuchipy>`_, the `Python Package Index
<https://pypi.org/project/kikuchipy/>`_ (``pip``), from source, or using the `HyperSpy
Bundle
<http://hyperspy.org/hyperspy-doc/current/user_guide/install.html#hyperspy-bundle>`_,
and supports Python >= 3.7. All alternatives are available on Windows, macOS, and Linux.

.. _install-with-anaconda:

With Anaconda
=============

To install with Anaconda, we recommend you install it in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
with the `Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html>`_. In
the Anaconda Prompt, terminal, or Command Prompt, create an environment and activate
it::

   $ conda create --name kp-env python=3.9
   $ conda activate kp-env

If you prefer a graphical interface to manage packages and environments, you can install
the `Anaconda distribution <https://docs.continuum.io/anaconda>`_ instead.

Installing::

    $ conda install kikuchipy --channel conda-forge

To update kikuchipy to the latest release::

    $ conda update kikuchipy

To install a specific version of kikuchipy (say version 0.3.4)::

    $ conda install kikuchipy==0.3.4

.. _install-with-pip:

With pip
========

kikuchipy is listed in the `Python Package Index <https://pypi.org/project/kikuchipy/>`_
(PyPI), and can therefore be installed with `pip <https://pip.pypa.io/en/stable>`_. To
do so, run the following in the Anaconda Prompt, terminal or Command Prompt::

    $ pip install kikuchipy

.. note::

    kikuchipy builds upon HyperSpy, which depends upon a number of libraries that
    usually need to be compiled. Installing kikuchipy with ``pip`` may therefore require
    some development tools.

To update kikuchipy to the latest release::

    $ pip install --upgrade kikuchipy

To install a specific version of kikuchipy (say version 0.3.4)::

    $ pip install kikuchipy==0.3.4

.. _install-from-source:

.. _install-with-hyperspy-bundle:

With the HyperSpy Bundle
========================

The easiest way to install kikuchipy is via the HyperSpy Bundle. See `HyperSpy's
documentation
<http://hyperspy.org/hyperspy-doc/current/user_guide/install.html#hyperspy-bundle>`_
for instructions.

.. warning::

    kikuchipy is updated more frequently than the HyperSpy Bundle, thus the installed
    version of kikuchipy will most likely not be the latest version available. See the
    `HyperSpy Bundle repository <https://github.com/hyperspy/hyperspy-bundle>`_ for how
    to update packages in the bundle.

From source
===========

To install kikuchipy from source, clone the repository from `GitHub
<https://github.com/pyxem/kikuchipy>`_, and install with ``pip``::

    $ git clone https://github.com/pyxem/kikuchipy.git
    $ cd kikuchipy
    $ pip install --editable .

See the :ref:`contributing guidelines <setting-up-a-development-installation>` for how
to set up a development installation and keep it up to date.

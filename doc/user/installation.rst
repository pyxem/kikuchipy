============
Installation
============

kikuchipy can be installed with `pip <https://pypi.org/project/kikuchipy/>`__,
`conda <https://anaconda.org/conda-forge/kikuchipy>`__, the `HyperSpy Bundle
<http://hyperspy.org/hyperspy-doc/current/user_guide/install.html#hyperspy-bundle>`__,
or from source, and supports Python >= 3.7. All alternatives are available on Windows,
macOS and Linux.

.. _install-with-pip:

With pip
========

kikuchipy is availabe from the Python Package Index (PyPI), and can therefore be
installed with `pip <https://pip.pypa.io/en/stable>`__. To install, run the following::

    pip install kikuchipy

To update kikuchipy to the latest release::

    pip install --upgrade kikuchipy

To install a specific version of kikuchipy (say version 0.5.8)::

    pip install kikuchipy==0.5.8

.. _optional-dependencies:

Optional dependencies
---------------------

Some dependencies are optional and available via the following selectors:

- ``viz``: 3D plot of master patterns using `pyvista <https://docs.pyvista.org/>`_

Installing optional dependencies::

    pip install kikuchipy[viz]

.. _install-with-anaconda:

With Anaconda
=============

To install with Anaconda, we recommend you install it in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
with the `Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html>`__.
To create an environment and activate it, run the following::

   conda create --name kp-env python=3.9
   conda activate kp-env

If you prefer a graphical interface to manage packages and environments, you can install
the `Anaconda distribution <https://docs.continuum.io/anaconda>`__ instead.

To install::

    conda install kikuchipy --channel conda-forge

To update kikuchipy to the latest release::

    conda update kikuchipy

To install a specific version of kikuchipy (say version 0.5.8)::

    conda install kikuchipy==0.5.8

.. _install-with-hyperspy-bundle:

With the HyperSpy Bundle
========================

kikuchipy is available in the HyperSpy Bundle. See `HyperSpy's documentation
<http://hyperspy.org/hyperspy-doc/current/user_guide/install.html#hyperspy-bundle>`__
for instructions.

.. warning::

    kikuchipy is updated more frequently than the HyperSpy Bundle, thus the installed
    version of kikuchipy will most likely not be the latest version available. See the
    `HyperSpy Bundle repository <https://github.com/hyperspy/hyperspy-bundle>`__ for how
    to update packages in the bundle.

.. _install-from-source:

From source
===========

To install kikuchipy from source, clone the repository from `GitHub
<https://github.com/pyxem/kikuchipy>`__, and install with ``pip``::

    git clone https://github.com/pyxem/kikuchipy.git
    cd kikuchipy
    pip install --editable .

See the :ref:`contributing guide <setting-up-a-development-installation>` for how
to set up a development installation and keep it up to date.

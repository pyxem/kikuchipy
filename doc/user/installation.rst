============
Installation
============

kikuchipy can be installed with `pip <https://pypi.org/project/kikuchipy/>`__,
`conda <https://anaconda.org/conda-forge/kikuchipy>`__, the
:ref:`hyperspy:hyperspy-bundle`, or from source, and supports Python >= 3.7.
All alternatives are available on Windows, macOS and Linux.

.. _install-with-pip:

With pip
========

kikuchipy is availabe from the Python Package Index (PyPI), and can therefore be
installed with `pip <https://pip.pypa.io/en/stable>`__.
To install, run the following::

    pip install kikuchipy

To update kikuchipy to the latest release::

    pip install --upgrade kikuchipy

To install a specific version of kikuchipy (say version 0.5.8)::

    pip install kikuchipy==0.5.8

.. _optional-dependencies:

Optional dependencies
---------------------

Some dependencies are optional and available via the following selectors:

======== ====================================== ==========================
Selector Package(s)                             Purpose
======== ====================================== ==========================
``viz``  `pyvista <https://docs.pyvista.org/>`_ 3D plot of master patterns
======== ====================================== ==========================

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

kikuchipy is available in the HyperSpy Bundle. See :ref:`hyperspy:hyperspy-bundle` for
instructions.

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

See the contributing guide for :ref:`setting-up-a-development-installation` and keeping
it up to date.

Dependencies
============

kikuchipy builds on the great work and effort of many people.
This is a list of explicit package dependencies (some are `Optional dependencies`_):

============================================================== =====================================================================
Package                                                        Purpose
============================================================== =====================================================================
`dask <https://docs.dask.org>`__                               Out-of-memory processing of data larger than RAM
`diffpy.structure <https://www.diffpy.org/diffpy.structure>`__ Handling of crystal structures
`diffsims <https://diffsims.readthedocs.io/en/latest>`__       Handling of reciprocal lattice vectors and structure factors
`hyperspy <https://hyperspy.org/hyperspy-doc/current>`__       Multi-dimensional data handling (EBSD class etc.)
`h5py <https://docs.h5py.org/en/stable>`__                     Read/write of HDF5 files
`imageio <https://imageio.readthedocs.io/en/stable>`__         Read image formats
`matplotlib <https://matplotlib.org/stable>`__                 Visualization
`numba <https://numba.pydata.org/numba-doc/latest/>`__         CPU acceleration
`numpy <https://numpy.org/doc/stable>`__                       Handling of N-dimensional arrays
`orix <https://orix.readthedocs.io/en/stable>`__               Handling and plotting of rotations and vectors using crystal symmetry
`pooch <https://www.fatiando.org/pooch/latest/>`__             Downloading and caching of datasets
`tqdm <https://tqdm.github.io/>`__                             Progressbars
`scikit-image <https://scikit-image.org/>`__                   Image processing like adaptive histogram equalization
`scikit-learn <https://scikit-learn.org/stable/>`__            Multivariate analysis
`scipy <https://docs.scipy.org/doc/scipy/>`__                  Optimization algorithms, filtering and more
============================================================== =====================================================================

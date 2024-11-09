============
Installation
============

kikuchipy can be installed with `pip <https://pypi.org/project/kikuchipy/>`__,
`conda <https://anaconda.org/conda-forge/kikuchipy>`__, the
:ref:`hyperspy:hyperspy-bundle`, or from source, and supports Python >= 3.10.
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

To install a specific version of kikuchipy (say version 0.8.5)::

    pip install kikuchipy==0.8.5

.. _install-with-anaconda:

With Anaconda
=============

To install with Anaconda, we recommend you install it in a
:doc:`conda environment <conda:user-guide/tasks/manage-environments>` with the
`Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html>`__.
To create an environment and activate it, run the following::

   conda create --name kp-env python=3.12
   conda activate kp-env

If you prefer a graphical interface to manage packages and environments, you can install
the `Anaconda distribution <https://docs.continuum.io/anaconda>`__ instead.

To install::

    conda install kikuchipy --channel conda-forge

To update kikuchipy to the latest release::

    conda update kikuchipy

To install a specific version of kikuchipy (say version 0.8.5)::

    conda install kikuchipy==0.8.5

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

.. _dependencies:

Dependencies
============

kikuchipy builds on the great work and effort of many people.
This is a list of core package dependencies:

* :doc:`dask<dask:index>`: Out-of-memory processing of data larger than RAM
* :doc:`diffpy.structure <diffpy.structure:index>`: Handling of crystal structures
* :doc:`diffsims <diffsims:index>`: Handling of reciprocal lattice vectors and structure
  factors
* :doc:`hyperspy <hyperspy:index>`: Multi-dimensional data handling (EBSD class etc.)
* :doc:`h5py <h5py:index>`: Read/write of HDF5 files
* :doc:`imageio <imageio:index>`: Read image formats
* `lazy_loader`_: Lazy loading of functions, classes, and modules
* :doc:`matplotlib <matplotlib:index>`: Visualization
* :doc:`numba <numba:index>`: CPU acceleration via just-in-time compilation
* :doc:`numpy <numpy:index>`: Handling of N-dimensional arrays
* :doc:`orix <orix:index>`: Handling of rotations and vectors using crystal symmetry
* :doc:`pooch <pooch:api/index>`: Downloading and caching of datasets
* `pyyaml <https://pyyaml.org/>`__: Parcing of YAML files
* :doc:`scikit-image <skimage:index>`: Image processing like adaptive histogram
  equalization
* :doc:`rosettasciio <rosettasciio:index>`: Read/write of some file formats
* `scikit-learn <https://scikit-learn.org/stable/>`__: Multivariate analysis
* :doc:`scipy <scipy:index>`: Optimization algorithms, filtering and more
* `tqdm <https://tqdm.github.io/>`__: Progressbars

.. _lazy_loader: https://scientific-python.org/specs/spec-0001/#lazy_loader

Some functionality requires optional dependencies:

* :doc:`pyebsdindex <pyebsdindex:index>`: Hough indexing. We recommend to install with
  optional GPU support via :doc:`pyopencl<pyopencl:index>` with
  ``pip install "pyebsdindex[gpu]""`` or ``conda install pyebsdindex -c conda-forge``.
* `nlopt <https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/>`__: Extra
  optimization algorithms used in EBSD orientation and/or projection center refinement.
  Installation from conda ``conda install nlopt -c conda-forge`` is recommended.
* :doc:`pyvista<pyvista:index>`: 3D plotting of master patterns.

Install all optional dependencies::

    pip install "kikuchipy[all]"

Note that this command will not install ``pyopencl``, which is required for GPU support
in ``pyebsdindex``. If the above command failed for some reason, you can try to install
each optional dependency individually.

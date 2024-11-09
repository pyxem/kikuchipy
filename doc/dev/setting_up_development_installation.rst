.. _setting-up-a-development-installation:

Setting up a development installation
=====================================

You need a `fork
<https://docs.github.com/en/get-started/quickstart/contributing-to-projects#about-forking>`__
of the `repository <https://github.com/pyxem/kikuchipy>`__ in order to make changes to
kikuchipy.

Make a local copy of your forked repository and change directories::

    git clone https://github.com/your-username/kikuchipy.git
    cd kikuchipy

Set the ``upstream`` remote to the main kikuchipy repository::

    git remote add upstream https://github.com/pyxem/kikuchipy.git

We recommend installing in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
with the `Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html>`__::

    conda create --name kp-dev
    conda activate kp-dev

Then, install the required dependencies while making the development version available
globally (in the ``conda`` environment)::

    pip install --editable ".[dev]"

This installs all necessary development dependencies, including those for running tests
and building documentation.
Further details are available in the :ref:`dependencies` section of the installation
guide.

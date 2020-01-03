=================
Install KikuchiPy
=================

You can install KikuchiPy with ``pip``, or by installing from source.

We recommend installing in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
with the `Miniconda distribution
<https://docs.conda.io/en/latest/miniconda.html>`_::

   conda create -n kikuchipy python=3.7
   conda activate kikuchipy

Pip
---

This installs both KikuchiPy and dependencies like HyperSpy, pyXem, and so on
that are necessary for different functionalities::

    pip install kikuchipy

Install from source
-------------------

To install KikuchiPy from source, clone the repository from `GitHub
<https://github.com/kikuchipy/kikuchipy>`_::

    git clone https://github.com/kikuchipy/kikuchipy.git
    cd kikuchipy
    pip install -e .

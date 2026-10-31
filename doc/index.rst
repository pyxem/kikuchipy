=================================
kikuchipy |release| documentation
=================================

kikuchipy is a library for processing, simulating and analyzing electron backscatter
diffraction (EBSD) patterns in Python, built on the tools for multi-dimensional data
analysis provided by the HyperSpy library.

See :doc:`user/licensing` for details on licensing.

.. toctree::
    :hidden:
    :titlesonly:

    user/index.rst
    reference/index.rst
    dev/index.rst
    changelog.rst


Installation
============

kikuchipy can be installed with `pip <https://pypi.org/project/kikuchipy>`__ or
`conda <https://anaconda.org/conda-forge/kikuchipy>`__:

.. tab-set::

    .. tab-item:: pip

        .. code-block:: bash

            pip install kikuchipy

    .. tab-item:: conda

        .. code-block:: bash

            conda install kikuchipy -c conda-forge

Further details are available in the :doc:`installation guide <user/installation>`.


Learning resources
==================

.. See: https://sphinx-design.readthedocs.io/en/furo-theme/grids.html
.. grid:: 2
    :gutter: 2

    .. grid-item-card::
        :link: tutorials/index
        :link-type: doc

        :octicon:`book;2em;sd-text-info` Tutorials
        ^^^

        In-depth guides for using kikuchipy.

    .. grid-item-card::
        :link: examples/index
        :link-type: doc

        :octicon:`zap;2em;sd-text-info` Examples
        ^^^

        Short recipies to common tasks using kikuchipy.

    .. grid-item-card::
        :link: reference/index
        :link-type: doc

        :octicon:`code;2em;sd-text-info` API reference
        ^^^

        Descriptions of all functions, modules, and objects in kikuchipy.

    .. grid-item-card::
        :link: dev/index
        :link-type: doc

        :octicon:`people;2em;sd-text-info` Contributing
        ^^^

        kikuchipy is a community project maintained for and by its users. There are many
        ways you can help!


Citing kikuchipy
================

If you are using kikuchipy in your research, please help our scientific visibility by
citing our work!

Paper:

.. code:: bibtex

    @article{aanes2026kikuchipy,
      author  = {{\AA}nes, H{\aa}kon W and Crout, Phillip and Lervik, Lars Andreas and Natlandsmyr, Ole and Bergh, Tina and Hjelen, Jarle and van Helvoort, Antonius TJ and Marthinsen, Knut},
      title   = {{kikuchipy: an open-source toolbox for analysis of EBSD patterns}},
      doi     = {10.1093/mam/ozag097},
      pages   = {Accepted},
      journal = {Microscopy and Microanalysis},
      year    = {2026},
    }

Pre-print:

.. code:: bibtex

    @article{aanes2026kikuchipy_arxiv,
      author  = {{\AA}nes, H{\aa}kon W and Crout, Phillip and Lervik, Lars Andreas and Natlandsmyr, Ole and Bergh, Tina and Hjelen, Jarle and van Helvoort, Antonius TJ and Marthinsen, Knut},
      title   = {{kikuchipy: an open-source toolbox for analysis of EBSD patterns}},
      doi     = {10.48550/arXiv.2605.25722},
      journal = {arXiv preprint arXiv:2605.25722},
      year    = {2026},
    }

Software DOI on Zenodo: https://doi.org/10.5281/zenodo.3597646.

=========
kikuchipy
=========

.. Launch binder
.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/pyxem/kikuchipy/HEAD
    :alt: Launch binder

.. Gitter chat
.. image:: https://badges.gitter.im/Join%20Chat.svg
    :target: https://gitter.im/pyxem/kikuchipy

.. Read the Docs
.. image:: https://readthedocs.org/projects/kikuchipy/badge/?version=latest
    :target: https://kikuchipy.org/en/latest/
    :alt: Documentation status

.. GitHub Actions
.. image:: https://github.com/pyxem/kikuchipy/workflows/build/badge.svg
    :target: https://github.com/pyxem/kikuchipy/actions
    :alt: Build status

.. Coveralls
.. image:: https://coveralls.io/repos/github/pyxem/kikuchipy/badge.svg?branch=develop
    :target: https://coveralls.io/github/pyxem/kikuchipy?branch=develop
    :alt: Coveralls status

.. PyPI version
.. image:: https://img.shields.io/pypi/v/kikuchipy.svg
    :target: https://pypi.python.org/pypi/kikuchipy
    :alt: PyPI version

.. Downloads per month
.. image:: https://pepy.tech/badge/kikuchipy/month
    :target: https://pepy.tech/project/kikuchipy
    :alt: Downloads per month

.. Zenodo DOI
.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.3597646.svg
    :target: https://doi.org/10.5281/zenodo.3597646
    :alt: DOI

kikuchipy is an open-source Python library for processing and analysis of electron
backscatter diffraction (EBSD) patterns. The library builds on the tools for
multi-dimensional data analysis provided by the HyperSpy library.

- **User guide and API reference**: https://kikuchipy.org. The guide consists of Jupyter
  Notebooks with many links to detailed explanations of the input parameters and output
  of functions and class methods (the API reference). The notebooks can be inspected
  statically on the web page or via `nbviewer`, downloaded and run locally, or run
  interactively in the browser after clicking the Binder link above and navigating to
  the `doc/user_guide` directory. We hope you find them useful!
- **License**: kikuchipy is released under the GPLv3+ license.
- **Cite**: If you find this project useful, please cite the DOI above.
- **Contribute**: Do you have a question or want to contribute? Great! Our
  `contributing guide <https://kikuchipy.org/en/latest/contributing.html>`_ explains how
  to do this.
- **Changelog**: The library is in an alpha stage, so there will be breaking changes
  with each release. Please see
  `the changelog <https://kikuchipy.org/en/latest/changelog.html>`_ for all
  developments.

.. toctree::
    :hidden:
    :caption: Getting started

    installation.rst

.. nbgallery::
    :caption: User guide

    user_guide/load_save_data.ipynb
    user_guide/reference_frames.ipynb
    user_guide/change_navigation_signal_shapes.ipynb
    user_guide/pattern_processing.ipynb
    user_guide/visualizing_patterns.ipynb
    user_guide/feature_maps.ipynb
    user_guide/virtual_backscatter_electron_imaging.ipynb
    user_guide/pattern_matching.ipynb
    user_guide/geometrical_ebsd_simulations.ipynb
    user_guide/multivariate_analysis.ipynb
    user_guide/metadata_structure.ipynb

.. toctree::
    :hidden:

    examples.rst

.. toctree::
    :hidden:
    :caption: Help & reference

    reference.rst
    bibliography.rst
    contributing.rst
    open_datasets.rst
    changelog.rst
    cite.rst
    related_projects.rst
    Code of Conduct <code_of_conduct.rst>

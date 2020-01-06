=========
KikuchiPy
=========

|travis-ci|_ |coveralls|_ |doc|_  |pypi_version|_  |doi|_ |black|_

.. |travis-ci| image:: https://api.travis-ci.org/kikuchipy/kikuchipy.svg?branch=master
.. _travis-ci: https://travis-ci.org/kikuchipy/kikuchipy

.. |coveralls| image:: https://coveralls.io/repos/github/kikuchipy/kikuchipy/badge.svg?branch=master
.. _coveralls: https://coveralls.io/github/kikuchipy/kikuchipy?branch=master

.. |doc| image:: https://readthedocs.org/projects/kikuchipy/badge/?version=latest
.. _doc: https://kikuchipy.readthedocs.io

.. |pypi_version| image:: http://img.shields.io/pypi/v/kikuchipy.svg?style=flat
.. _pypi_version: https://pypi.python.org/pypi/kikuchipy

.. |doi| image:: https://zenodo.org/badge/doi/10.5281/zenodo.3597646.svg
.. _doi: https://doi.org/10.5281/zenodo.3597646

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
.. _black: https://github.com/psf/black

*KikuchiPy is an open-source Python library for processing and analysis of
electron backscatter diffraction (EBSD) patterns.*

The library builds upon the tools for multi-dimensional data analysis provided
by the `HyperSpy <https://hyperspy.org>`_ library. This means that the EBSD
class, which has several common methods for processing of EBSD patterns, also
inherits all relevant methods from HyperSpy's Signal2D and `Signal
<https://hyperspy.org/hyperspy-doc/current/user_guide/tools.html>`_ classes.

Note that the project is in an alpha stage, and there will likely be breaking
changes with each release.

KikuchiPy is released under the GPL v3 license.

.. toctree::
    :hidden:
    :caption: Getting started

    install.rst

.. toctree::
    :hidden:
    :caption: User guide

    load_save_patterns.rst
    change_scan_pattern_size.rst
    background_correction.rst
    visualizing_patterns.rst
    virtual_backscatter_electron_imaging.rst
    multivariate_analysis.rst
    metadata.rst
    reference.rst

.. toctree::
    :hidden:
    :caption: Help & reference

    contributing.rst
    open_datasets.rst
    changelog.rst
    cite.rst
    related_projects.rst
    Code of Conduct <code_of_conduct.rst>

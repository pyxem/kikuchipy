.. Travis CI
.. image:: https://travis-ci.com/kikuchipy/kikuchipy.svg?branch=master
    :target: https://travis-ci.com/kikuchipy/kikuchipy
    :alt: Build status

.. Coveralls
.. image:: https://img.shields.io/coveralls/github/kikuchipy/kikuchipy.svg
    :target: https://coveralls.io/github/kikuchipy/kikuchipy?branch=master
    :alt: Coveralls status

.. Read the Docs
.. image:: https://readthedocs.org/projects/kikuchipy/badge/?version=latest
    :target: https://kikuchipy.org/en/latest/
    :alt: Documentation status

.. PyPI version
.. image:: https://img.shields.io/pypi/v/kikuchipy.svg
    :target: https://pypi.python.org/pypi/kikuchipy
    :alt: PyPI version

.. conda-forge version
.. image:: https://img.shields.io/conda/vn/conda-forge/kikuchipy
    :target: https://anaconda.org/conda-forge/kikuchipy
    :alt: conda-forge version

.. Zenodo DOI
.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.3597646.svg
    :target: https://doi.org/10.5281/zenodo.3597646
    :alt: DOI

*kikuchipy is an open-source Python library for processing and analysis of
electron backscatter diffraction (EBSD) patterns.*

The library builds on the tools for multi-dimensional data analysis provided
by the `HyperSpy <https://hyperspy.org>`_ library. This means that the EBSD
class, which has several common methods for processing of EBSD patterns, also
inherits all relevant methods from HyperSpy's Signal2D and `Signal
<https://hyperspy.org/hyperspy-doc/current/user_guide/tools.html>`_ classes.

An effort is made to keep memory usage in check and enable scalability by using
the `Dask <https://dask.org>`_ library for pattern processing.

The project is in an alpha stage, and there will likely be breaking changes with
each release.

kikuchipy is released under the GPL v3 license.

User guide
----------

Installation instructions, a user guide and the full API reference is available
via Read the Docs at https://kikuchipy.org.

Contributing
------------

Everyone is welcome to contribute. Please read our `contributor guide
<https://kikuchipy.org/en/latest/contributing.html>`_ to get started!

Code of Conduct
---------------

kikuchipy has a `Code of Conduct
<https://kikuchipy.org/en/latest/code_of_conduct.html>`_ that should be honoured
by everyone who participates in the kikuchipy community.

Cite
----

If analysis using kikuchipy forms a part of published work, please consider
recognizing the code development by citing the DOI above.

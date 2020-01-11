.. Travis CI
.. image:: https://travis-ci.org/kikuchipy/kikuchipy.svg?branch=master
    :target: https://travis-ci.org/kikuchipy/kikuchipy
    :alt: Build status

.. Coveralls
.. image:: https://img.shields.io/coveralls/github/kikuchipy/kikuchipy.svg
    :target: https://coveralls.io/github/kikuchipy/kikuchipy?branch=master
    :alt: Coveralls status

.. Read the Docs
.. image:: https://readthedocs.org/projects/kikuchipy/badge/?version=latest
    :target: https://kikuchipy.readthedocs.io/en/latest/?badge=latest
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

User guide
----------

Installation instructions, a user guide and the full API reference is available
`here <https://kikuchipy.readthedocs.io>`_ via Read the Docs.

Contributing
------------

Everyone is welcome to contribute. Please read our `contributor guide
<https://kikuchipy.readthedocs.io/en/latest/contributing.html>`_ to get started!

Code of Conduct
---------------

KikuchiPy has a `Code of Conduct
<https://kikuchipy.readthedocs.io/en/latest/code_of_conduct.html>`_
that should be honoured by everyone who participates in the KikuchiPy community.

Cite
----

If analysis using KikuchiPy forms a part of published work, please consider
recognizing the code development by citing the DOI above.

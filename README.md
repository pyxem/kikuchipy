[![Build Status](https://api.travis-ci.org/kikuchipy/kikuchipy.svg?branch=master)](https://travis-ci.org/kikuchipy/kikuchipy)
[![Coverage Status](https://coveralls.io/repos/github/kikuchipy/kikuchipy/badge.svg?branch=master)](https://coveralls.io/github/kikuchipy/kikuchipy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/kikuchipy/badge/?version=latest)](https://kikuchipy.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/kikuchipy.svg?style=flat)](https://pypi.org/project/kikuchipy/)
[![DOI](https://zenodo.org/badge/160169099.svg?)](https://zenodo.org/badge/latestdoi/160169099)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black)

*KikuchiPy is an open-source Python library for processing and analysis of
electron backscatter diffraction (EBSD) patterns.*

The library builds upon the tools for multi-dimensional data analysis provided
by the [HyperSpy](https://hyperspy.org/) library. This means that the `EBSD`
class, which has several common methods for processing of EBSD patterns, also
inherits all relevant methods from HyperSpy's `Signal2D` and `Signal` classes.

Note that the project is in an alpha stage, and there will likely be breaking
changes with each release.

KikuchiPy is released under the GPL v3 license.

### User guide

Installation instructions, a user guide and the full API reference is available
[here](https://kikuchipy.readthedocs.io) via Read the Docs.

### Contributing

Everyone is welcome to contribute. Please read our
[contributor guide](https://kikuchipy.readthedocs.io/en/latest/contributing.html)
to get started!

### Cite

If analysis using KikuchiPy forms a part of published work, please consider
recognizing the code development by citing the DOI above.

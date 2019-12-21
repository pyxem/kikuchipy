[![Build Status](https://api.travis-ci.org/kikuchipy/kikuchipy.svg?branch=master)](https://travis-ci.org/kikuchipy/kikuchipy) [![Coverage Status](https://coveralls.io/repos/github/kikuchipy/kikuchipy/badge.svg?branch=master)](https://coveralls.io/github/kikuchipy/kikuchipy?branch=master) [![Documentation Status](https://readthedocs.org/projects/kikuchipy/badge/?version=latest)](https://kikuchipy.readthedocs.io/en/latest/?badge=latest)

*KikuchiPy is an open-source Python library for processing and analysis of
electron backscatter diffraction (EBSD) patterns.*

The library builds upon the tools for multi-dimensional data analysis provided
by the [HyperSpy](https://hyperspy.org/) library. This means that the `EBSD`
class, which has several common methods for processing of EBSD patterns, also
inherits all relevant methods from HyperSpy's `Signal2D` and `Signal` classes.

KikuchiPy is released under the GPL v3 license.

#### Installation
```bash
$ conda create --name kikuchi python=3
$ conda activate kikuchi
$ conda install hyperspy pyxem --channel conda-forge
$ pip install https://github.com/kikuchipy/kikuchipy/archive/master.zip
```

#### Use
```python
import kikuchipy as kp
s = kp.load('/some/data.h5')
s.static_background_correction()
s.dynamic_background_correction()
s.adaptive_histogram_equalization()
s.save('/some/processed_data.h5')  # E.g. for subsequent indexing with EMsoft
```

#### Supported EBSD formats
* NORDIF .dat binary files (read/write)
* HyperSpy's .hspy files (read/write)
* h5ebsd files, specifically EDAX's and Bruker's formats (read) and
KikuchiPy's own format (read/write)

Readers for more formats will be added.

#### Contribute
Everyone is welcome to contribute, if it is by raising issues in the
[issue tracker](https://github.com/kikuchipy/kikuchipy/issues) or by
contributing code in [pull requests](https://github.com/kikuchipy/kikuchipy/pulls).

#### Cite
If analysis using KikuchiPy forms a part of published work please consider
recognizing the code development, by for now citing HyperSpy.

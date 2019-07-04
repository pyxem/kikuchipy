KikuchiPy
------------
KikuchiPy is an open-source Python library for processing of electron
backscatter diffraction (EBSD) patterns. It builds upon the tools for
multi-dimensional data analysis provided by the HyperSpy library for treatment
of experimental EBSD data.

KikuchiPy is released under the GPL v3 license.

Installation
------------
KikuchiPy requires Python 3 and conda – we suggest using the Python 3 version of [Miniconda](https://conda.io/miniconda.html) and creating a new environment for KikuchiPy using the following commands in the Anaconda Prompt within the top directory with the `setup.py` file:

```bash
$ conda create -n kikuchi python=3.7
$ conda activate kikuchi
$ conda install -c conda-forge pyxem
$ python setup.py install
```

KikuchiPy depends on pyXem. pyXem depends on DiffPy, and while KikuchiPy does not depend on DiffPy, pyXem must be installed first for the installation to be successful.

Use
---
Jupyter Notebooks explaining typical workflows will be made available in a separate repository in the near future.

Supported EBSD formats
----------------------
Read/write:
* NORDIF .dat binary files
* HyperSpy's .hspy files

Read:
* EDAX TSL HDF5 format

With plans to support:
* EMEBSD (EMsoft's HDF5 format)
* EDAX TSL up1 and up2 formats

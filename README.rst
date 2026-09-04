|logo|
======

.. |logo| image:: https://raw.githubusercontent.com/pyxem/kikuchipy/develop/doc/_static/logo/plasma_banner.png
   :width: 50%
   :target: https://kikuchipy.org

kikuchipy [ki-ko-chi-pai] is a library for processing, simulating, and indexing of
electron backscatter diffraction (EBSD) patterns in Python.
It is built on the tools for multi-dimensional data analysis provided by the HyperSpy
library.

.. |pypi_version| image:: https://img.shields.io/pypi/v/kikuchipy.svg?logo=python&logoColor=white
   :target: https://pypi.python.org/pypi/kikuchipy

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/kikuchipy.svg?logo=conda-forge&logoColor=white
   :target: https://anaconda.org/conda-forge/kikuchipy

.. |tests_status| image:: https://github.com/pyxem/kikuchipy/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/pyxem/kikuchipy/actions/workflows/tests.yml

.. |python| image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/

.. |coverage| image:: https://codecov.io/github/pyxem/kikuchipy/graph/badge.svg?token=TLRI1M0LBB
   :target: https://codecov.io/github/pyxem/kikuchipy

.. |pypi_downloads| image:: https://img.shields.io/pypi/dm/kikuchipy.svg?label=pypi%20downloads
   :target: https://pypi.org/project/kikuchipy

.. |conda_downloads| image:: https://img.shields.io/conda/dn/conda-forge/kikuchipy.svg?label=conda%20downloads
   :target: https://anaconda.org/conda-forge/kikuchipy

.. |arxiv_doi| image:: https://img.shields.io/badge/DOI-10.48550/arXiv.2605.25722-blue
   :target: https://doi.org/10.48550/arXiv.2605.25722

.. |zenodo_doi| image:: https://zenodo.org/badge/doi/10.5281/zenodo.3597646.svg
   :target: https://doi.org/10.5281/zenodo.3597646

.. |GPLv3| image:: https://img.shields.io/badge/license-GPLv3%2B-blue
   :target: https://opensource.org/license/GPL-3.0

.. |BSD3| image:: https://img.shields.io/badge/license_(select_components)-BSD--3--Clause-blue
   :target: https://opensource.org/license/BSD-3-Clause

.. |GH-discuss| image:: https://img.shields.io/badge/GitHub-Discussions-green?logo=github
   :target: https://github.com/pyxem/kikuchipy/discussions

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyxem/kikuchipy/HEAD

.. |docs| image:: https://readthedocs.org/projects/kikuchipy/badge/?version=latest
   :target: https://kikuchipy.org/en/latest

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

+----------------------+------------------------------------------------+
| Deployment           | |pypi_version| |conda|                         |
+----------------------+------------------------------------------------+
| Build status         | |tests_status| |docs| |python|                 |
+----------------------+------------------------------------------------+
| Metrics              | |coverage|                                     |
+----------------------+------------------------------------------------+
| Activity             | |pypi_downloads| |conda_downloads|             |
+----------------------+------------------------------------------------+
| Citation             | |arxiv_doi| |zenodo_doi|                       |
+----------------------+------------------------------------------------+
| License              | |GPLv3| |BSD3|                                 |
+----------------------+------------------------------------------------+
| Community            | |GH-discuss|                                   |
+----------------------+------------------------------------------------+
| Formatter            | |black|                                        |
+----------------------+------------------------------------------------+

kikuchipy is licensed under the `GNU General Public License v3+
<https://opensource.org/license/GPL-3.0>`__.
Select components are individually licensed under the `BSD 3-Clause License
<https://opensource.org/license/BSD-3-Clause>`__, allowing reuse *of those components
only* in both GPL and non-GPL projects.


Documentation
-------------

Refer to the `documentation <https://kikuchipy.org>`__ for detailed installation
instructions, a user guide, and the
`changelog <https://kikuchipy.org/en/stable/changelog.html>`__.


Installation
------------

kikuchipy can be installed with ``pip``::

    pip install kikuchipy

or ``conda``::

    conda install kikuchipy -c conda-forge

You can also visit `PyPI <https://pypi.org/project/kikuchipy>`__,
`Anaconda <https://anaconda.org/conda-forge/kikuchipy>`__, or
`GitHub <https://github.com/pyxem/kikuchipy>`__ to download the source.

Further details are available in the
`installation guide <https://kikuchipy.org/en/stable/user/installation.html>`__.


Citing kikuchipy
----------------

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

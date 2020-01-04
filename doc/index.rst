=========
KikuchiPy
=========

*KikuchiPy is an open-source Python library for processing and analysis of
electron backscatter diffraction (EBSD) patterns.*

The library builds upon the tools for multi-dimensional data analysis provided
by the `HyperSpy <https://hyperspy.org>`_ library. This means that the EBSD
class, which has several common methods for processing of EBSD patterns, also
inherits all relevant methods from HyperSpy's Signal2D and `Signal
<https://hyperspy.org/hyperspy-doc/current/user_guide/tools.html>`_ classes.

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

Acknowledgement
===============

Initial work from Håkon Wiik Ånes was funded by the NAPIC project (NTNU
Aluminium Product Innovation Center).

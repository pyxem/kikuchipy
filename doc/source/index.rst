=========
KikuchiPy
=========

*KikuchiPy is an open-source Python library for processing of electron
backscatter diffraction (EBSD) patterns.*

Processing builds upon the tools for multi-dimensional data analysis provided
by the `HyperSpy`_ library. This means that the EBSD class, which has several
common methods for processing of EBSD patterns, also inherits all relevant
methods from HyperSpy's `Signal2D`_ and `Signal`_ classes.

.. _`HyperSpy`: https://hyperspy.org
.. _`Signal2D`: https://hyperspy.org/hyperspy-doc/current/user_guide/signal2d.html
.. _`Signal`: https://hyperspy.org/hyperspy-doc/current/user_guide/tools.html

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting started

    install.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: User guide

    load_save_patterns.rst
    change_scan_pattern_size.rst
    background_corrections.rst
    visualizing_patterns.rst
    virtual_forward_scatter_detector.rst
    metadata.rst
    bibliography.rst
    reference.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Help & reference

    develop.rst
    open_datasets.rst
    changelog.rst
    cite.rst

Acknowledgement
===============

Initial work from Håkon Wiik Ånes funded by the NAPIC project (NTNU Aluminium
Product Innovation Center).

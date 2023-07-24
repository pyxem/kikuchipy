.. _api:

=============
API reference
=============

**Release**: |version|

**Date**: |today|

This reference manual describes the public functions, modules, and objects in kikuchipy.
Many of the descriptions include brief examples. For learning how to use kikuchipy, see
the :doc:`/examples/index` or :doc:`/tutorials/index`.

.. caution::

    kikuchipy is in continuous development, meaning that some breaking changes and
    changes to this reference are likely with each release.

kikuchipy's import structure is designed to feel familiar to HyperSpy users. It is
recommended to import functionality from the below list of functions and modules like
this:

.. autolink-skip::
.. code-block:: python

    >>> import kikuchipy as kp
    >>> s = kp.data.nickel_ebsd_small()
    >>> s
    <EBSD, title: patterns Scan 1, dimensions: (3, 3|60, 60)>

.. currentmodule:: kikuchipy

.. rubric:: Functions

.. autosummary::
    :toctree: generated

    load
    set_log_level

.. rubric:: Modules

.. autosummary::
    :toctree: generated
    :template: custom-module-template.rst

    data
    detectors
    draw
    filters
    imaging
    indexing
    io
    pattern
    signals
    simulations

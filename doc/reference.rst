===================
KikuchiPy reference
===================

This reference manual details all EBSD class methods, utility functions and
readers of EBSD patterns included in KikuchiPy, as generated from their
docstrings. For learning how to use KikuchiPy, see the user guide pages in the
sidebar.

EBSD
====

While some methods listed here are only available to
:py:class:`~kikuchipy.signals.ebsd.LazyEBSD` objects, all methods available to
:py:class:`~kikuchipy.signals.ebsd.EBSD` objects, apart from
:py:meth:`~kikuchipy.signals.ebsd.EBSD.as_lazy`, are also available to
:py:class:`~kikuchipy.signals.ebsd.LazyEBSD` objects.

.. autoclass:: kikuchipy.signals.ebsd.EBSD
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: kikuchipy.signals.ebsd.LazyEBSD
    :members:
    :undoc-members:
    :show-inheritance:

Utilities
=========

Experimental utilities
----------------------
.. automodule:: kikuchipy.util.experimental
    :members:
    :undoc-members:

Input/output utilities
----------------------
.. automodule:: kikuchipy.io._io
    :members:
    :undoc-members:

.. automodule:: kikuchipy.util.io
    :members:
    :undoc-members:

Input/output plugins
====================

These plugin functions import patterns and parameters from vendor file formats
into :py:class:`~kikuchipy.signals.ebsd.EBSD` (or
:py:class:`~kikuchipy.signals.ebsd.LazyEBSD` if loading lazily) objects.

h5ebsd
------
.. automodule:: kikuchipy.io.plugins.h5ebsd
    :members:
    :undoc-members:

NORDIF
------
.. automodule:: kikuchipy.io.plugins.nordif
    :members:
    :undoc-members:

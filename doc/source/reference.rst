===================
KikuchiPy reference
===================

This reference manual details all EBSD class methods, utility functions and
readers of EBSD patterns included in KikuchiPy, as generated from their
docstrings. For learning how to use KikuchiPy, see the user guide pages in the
sidebar.

.. note::
    Private functions that start with an underscore are listed in the reference
    for completeness. However, they may be removed, moved or their input
    parameters and behaviour may change between minor package updates. You
    should take care if you use them directly.

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
    :private-members:
    :show-inheritance:

.. autoclass:: kikuchipy.signals.ebsd.LazyEBSD
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:

Utilities
=========

These functions are included here for completeness and are not intended for
direct use.

Dask utilities
--------------
.. automodule:: kikuchipy.util.dask
    :members:
    :undoc-members:
    :private-members:

Decomposition utilities
-----------------------
.. automodule:: kikuchipy.util.decomposition
    :members:
    :undoc-members:
    :private-members:

Experimental utilities
----------------------
.. automodule:: kikuchipy.util.experimental
    :members:
    :undoc-members:
    :private-members:

Input/output utilities
----------------------
.. automodule:: kikuchipy.io._io
    :members:
    :undoc-members:
    :private-members:

.. automodule:: kikuchipy.util.io
    :members:
    :undoc-members:
    :private-members:

Other utilities
---------------
.. automodule:: kikuchipy.util.general
    :members:
    :undoc-members:
    :private-members:

Phase utilities
---------------
.. automodule:: kikuchipy.util.phase
    :members:
    :undoc-members:
    :private-members:

Input/Output plugins
====================

h5ebsd
------
.. automodule:: kikuchipy.io.plugins.h5ebsd
    :members:
    :undoc-members:
    :private-members:

NORDIF
------
.. automodule:: kikuchipy.io.plugins.nordif
    :members:
    :undoc-members:
    :private-members:

===================
KikuchiPy reference
===================

This reference manual details all EBSD class methods and utility functions
included in KikuchiPy, as generated from their docstrings. For learning how to
use KikuchiPy, see the user guide pages in the sidebar.

Note that while also private functions, i.e. functions that starts with an
underscore, are listed, their input parameters and behaviours may change between
minor package updates.

EBSD class and methods
======================

While some methods listed here are only available to LazyEBSD objects, almost
all methods available to EBSD objects, apart from
:py:meth:`~kikuchipy.signals.ebsd.EBSD.as_lazy()`, are also available to
LazyEBSD objects.

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

Utility functions
=================

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

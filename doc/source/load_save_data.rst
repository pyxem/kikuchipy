=======================
Loading and saving data
=======================

EBSD patterns from supported formats are loaded using the
:py:func:`~kikuchipy.io._io.load` command. For example, to load the first scan from an
EDAX TSL's HDF5 format we type:

.. code-block:: python

    >>> s = kp.load('patterns.h5ebsd')

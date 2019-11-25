======================
Load and save patterns
======================

.. _load-patterns-from-file:

Load patterns
=============

.. _from-file:

From file
-----------

KikuchiPy can read and write EBSD patterns from/to multiple formats (see
:ref:`supported-ebsd-formats`). To load patterns from file use the
:py:func:`~kikuchipy.io._io.load` function. For example, to load the first scan
from an EDAX TSL .h5 file into memory:

.. code-block:: python

    >>> s = kp.load('patterns.h5')

All file readers support accessing the data without loading it into memory,
which can be useful when processing large scans:

.. code-block:: python

    >>> s = kp.load('patterns.h5', lazy=True)

Note that this also means that processing will take longer, and some care needs
to be taken when performing the processing. See the relevant
`HyperSpy documentation`_ for information on how to do this.

.. _`HyperSpy documentation`: http://hyperspy.org/hyperspy-doc/current/user_guide/big_data.html

To visualise the data:

.. code-block:: python

    >>> s.plot()

KikuchiPy tries to read extra information about the EBSD scan and stores
everything it can read in the ``original_metadata`` attribute. Also, some
information may be stored in a standard location in the ``metadata`` attribute
where it can be used by some routines. The number of patterns in horizontal and
vertical direction, pattern size in pixels, scan step size and detector pixel
size is stored in the ``axes_manager`` attribute. To access this information:

.. code-block:: python

    >>> s.metadata
    >>> s.original_metadata
    >>> s.axes_manager

This information can be modified directly, while information in ``metadata`` and
``axes_manager`` can also be modified by the
:py:class:`~kikuchipy.signals.ebsd.EBSD` class methods
:py:class:`~kikuchipy.signals.ebsd.EBSD.set_experimental_parameters`,
:py:class:`~kikuchipy.signals.ebsd.EBSD.set_phase_parameters`,
:py:class:`~kikuchipy.signals.ebsd.EBSD.set_scan_calibration` and
:py:class:`~kikuchipy.signals.ebsd.EBSD.set_detector_calibration`. For example,
to set or change the accelerating voltage and horizontal pattern centre
coordinate:

.. code-block:: python

    >>> s.set_experimental_parameters(beam_energy=15, xpc=0.5073)

.. _from-numpy-array:

From NumPy array
----------------

An :py:class:`~kikuchipy.signals.ebsd.EBSD` object can also be created directly
from a :py:class:`numpy.ndarray`. To create a data set of (60 x 60) pixel
patterns in a (10 x 20) grid, i.e. 10 and 20 patterns in the horizontal and
vertical scan directions respectively, of random intensities:

.. code-block:: python

    >>> import numpy as np
    >>> import kikuchipy as kp
    >>> s = kp.signals.EBSD(np.random.random((20, 10, 60, 60)))
    >>> s
    <EBSD, title: , dimensions: (10, 20|60, 60)>

.. _from-dask-array:

From Dask array
---------------

When processing large scans it is useful to load data lazily, e.g. with the
`Dask library`_. This can be done when reading patterns
:ref:`from a file <from-file>` by setting ``lazy=True`` when using
:py:func:`~kikuchipy.io._io.load`, or directly from a
:py:class:`dask.array.Array`:

.. _`Dask library`: https://docs.dask.org/en/latest/

.. code-block:: python

    >>> import dask.array as da
    >>> import kikuchipy as kp
    >>> s = kp.signals.LazyEBSD(da.random.random((20, 10, 60, 60), chunks=(2, 2, 60, 60)))
    >>> s
    <LazyEBSD, title: , dimensions: (10, 20|60, 60)>

.. _from-hyperspy-signal:

From HyperSpy signal
--------------------

HyperSpy provides the method
:py:meth:`~hyperspy.signal.BaseSignal.set_signal_type` to change between
:py:class:`~hyperspy.signal.BaseSignal` subclasses, of which
:py:class:`~kikuchipy.signals.ebsd.EBSD` is one. To create an
:py:class:`~kikuchipy.signals.ebsd.EBSD` object from a
:py:class:`~hyperspy.signals.Signal2D` object:

.. code-block:: python

    >>> import numpy as np
    >>> import hyperspy.api as hs
    >>> import kikuchipy as kp
    >>> s = hs.signals.Signal2D(np.random.random((20, 10, 60, 60)))
    >>> s
    <Signal2D, title: , dimensions: (10, 20|60, 60)>
    >>> s.set_signal_type('EBSD')
    >>> s
    <EBSD, title: , dimensions: (10, 20|60, 60)>

.. _save-patterns:

Save patterns
=============

To save patterns to file use the :py:meth:`~kikuchipy.signals.ebsd.EBSD.save`
method. For example, to save an :py:class:`~kikuchipy.signals.ebsd.EBSD` object
``s`` in an HDF5 file, with file name `patterns.h5`, in our default
:ref:`h5ebsd-format` format:

.. code-block:: python

    >>> s.save('patterns')

.. danger::

    If you want to overwrite an existing h5ebsd file:

    .. code-block:: python

        >>> s.save('patterns.dat', overwrite=True)

If you want to save patterns in NORDIF's binary .dat format instead:

.. code-block:: python

    >>> s.save('patterns.dat')

.. _supported-ebsd-formats:

Supported EBSD formats
======================

Currently, KikuchiPy has readers and writers for the following file formats:

.. _supported-formats-table:

.. table::

    +--------------------+------+-------+
    | Format             | Read | Write |
    +====================+======+=======+
    | Bruker Nano h5ebsd | Yes  | No    |
    +--------------------+------+-------+
    | EDAX TSL h5ebsd    | Yes  | No    |
    +--------------------+------+-------+
    | KikuchiPy h5ebsd   | Yes  | Yes   |
    +--------------------+------+-------+
    | NORDIF binary      | Yes  | Yes   |
    +--------------------+------+-------+

.. note::

    If you want to process your patterns with KikuchiPy, but use an unsupported
    EBSD vendor software, or if you want to write your processed patterns to a
    vendor format that does not support writing, please request the feature
    in our `code repository <https://github.com/kikuchipy/kikuchipy/issues>`_.

.. _h5ebsd-format:

h5ebsd
------

The h5ebsd format :ref:`[Jackson2014] <[Jackson2014]>` is based on the `HDF5
open standard <http://www.hdfgroup.org/HDF5/>`_ (Hierarchical Data Format
version 5). When reading an HDF5 file with extension ``.h5``, ``.hdf5`` or
``.h5ebsd``, the correct reader is determined from the file. Supported h5ebsd
formats are listed in the :ref:`table above <supported-formats-table>`.

If an h5ebsd file contains multiple scans, as many scans as desirable can be
read from the file. For example, if the file contains three scans with names
``Scan 1``, ``Scan 4`` and ``Scan 6``:

.. code-block:: python

    >>> s1, s4, s6 = kp.load('patterns.h5', scans=[1, 4, 6])

If only ``Scan 4`` is to be read, ``scans=4`` can be passed. The ``scans``
parameter is unnecessary if only ``Scan 1`` is to be read since reading the
first scan in the file is the default behaviour.

So far, only :ref:`saving patterns <save-patterns>` to KikuchiPy's own h5ebsd
format is supported. It is possible to write a new scan with a new scan number
to an existing h5ebsd file in the KikuchiPy format, e.g. one containing only
``Scan 1``, by passing:

.. code-block:: python

    >>> s.save('patterns.h5', add_scan=True, scan_number=2)

.. _nordif-format:

NORDIF binary
-------------

This is NORDIF's binary file format.

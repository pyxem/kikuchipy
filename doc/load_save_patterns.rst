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
:func:`~kikuchipy.io._io.load` function. For example, to load the first scan
from an EDAX TSL .h5 file into memory:

.. code-block:: python

    >>> s = kp.load('patterns.h5')

All file readers support accessing the data without loading it into memory,
which can be useful when processing large scans:

.. code-block:: python

    >>> s = kp.load('patterns.h5', lazy=True)

When loading lazily patterns are processed chunk by chunk which in many cases
leads to longer processing times, so processing should be done with some care.
See the relevant `HyperSpy user guide
<http://hyperspy.org/hyperspy-doc/current/user_guide/big_data.html>`_ for
information on how to do this.

To visualise the data:

.. code-block:: python

    >>> s.plot()

Upon loading KikuchiPy tries to read all scan information from the file and
stores everything it can read in the ``original_metadata`` attribute. Also, some
information may be stored in a standard location in the ``metadata`` attribute
where it can be used by some routines. The number of patterns in horizontal and
vertical direction, pattern size in pixels, scan step size and detector pixel
size is stored in the ``axes_manager`` attribute. To access this information:

.. code-block:: python

    >>> s.metadata
    >>> s.original_metadata
    >>> s.axes_manager

This information can be modified directly, and information in ``metadata`` and
``axes_manager`` can also be modified by the
:class:`~kikuchipy.signals.ebsd.EBSD` class methods
:meth:`~kikuchipy.signals.ebsd.EBSD.set_experimental_parameters`,
:meth:`~kikuchipy.signals.ebsd.EBSD.set_phase_parameters`,
:meth:`~kikuchipy.signals.ebsd.EBSD.set_scan_calibration` and
:meth:`~kikuchipy.signals.ebsd.EBSD.set_detector_calibration`. For example, to
set or change the accelerating voltage, horizontal pattern centre coordinate and
static background pattern (stored as a :class:`numpy.ndarray`):

.. code-block:: python

    >>> s.set_experimental_parameters(
            beam_energy=15, xpc=0.5073, static_bg=static_bg)

.. _from-numpy-array:

From NumPy array
----------------

An :class:`~kikuchipy.signals.ebsd.EBSD` object can also be created directly
from a :class:`numpy.ndarray`. To create a data set of (60 x 60) pixel patterns
in a (10 x 20) grid, i.e. 10 and 20 patterns in the horizontal and vertical scan
directions respectively, of random intensities:

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
`Dask library <https://docs.dask.org/en/latest/>`_. This can be done when
reading patterns :ref:`from a file <from-file>` by setting ``lazy=True`` when
using :func:`~kikuchipy.io._io.load`, or directly from a
:py:class:`dask.array.Array`:

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
:meth:`~hyperspy.signal.BaseSignal.set_signal_type` to change between
:class:`~hyperspy.signal.BaseSignal` subclasses, of which
:class:`~kikuchipy.signals.ebsd.EBSD` is one. To create an
:class:`~kikuchipy.signals.ebsd.EBSD` object from a
:class:`~hyperspy.signals.Signal2D` object:

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

To save patterns to file use the :meth:`~kikuchipy.signals.ebsd.EBSD.save`
method. For example, to save an :class:`~kikuchipy.signals.ebsd.EBSD` object
``s`` in an HDF5 file, with file name `patterns.h5`, in our default
:ref:`h5ebsd-format` format:

.. code-block:: python

    >>> s.save('patterns')

.. danger::

    If you want to overwrite an existing file:

    .. code-block:: python

        >>> s.save('patterns.h5', overwrite=True)

If you want to save patterns in NORDIF's binary .dat format instead:

.. code-block:: python

    >>> s.save('patterns.dat')

.. note::

    To save results from statistical decomposition (machine learning) of
    patterns to file see the section `Saving and loading results
    <http://hyperspy.org/hyperspy-doc/current/user_guide/mva.html#saving-and-
    loading-results>`_ in HyperSpy's user guide. Note that the file extension
    ``.hspy`` must be used upon saving, ``s.save('patterns.hspy')``, as the
    default extension in KikuchiPy, ``.h5``, yields a KikuchiPy h5ebsd file. The
    saved patterns can be then reloaded using HyperSpy's
    :func:`~hyperspy.io.load` function followed by ``set_signal_type('EBSD')``
    :ref:`as explained above <from-hyperspy-signal>`.

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
    vendor format that does not support writing, please request this feature
    in our `code repository <https://github.com/kikuchipy/kikuchipy/issues>`_.

.. _h5ebsd-format:

h5ebsd
------

The h5ebsd format [Jackson2014]_ is based on the `HDF5 open standard
<http://www.hdfgroup.org/HDF5/>`_ (Hierarchical Data Format version 5). HDF5
files can be read and edited using e.g. the HDF Group's reader `HDFView
<https://www.hdfgroup.org/downloads/hdfview/>`_ or the Python packed used here,
`h5py <http://docs.h5py.org/en/stable/>`_. Upon loading an HDF5 file with
extension ``.h5``, ``.hdf5`` or ``.h5ebsd``, the correct reader is determined
from the file. Supported h5ebsd formats are listed in the :ref:`table above
<supported-formats-table>`.

If an h5ebsd file contains multiple scans, as many scans as desirable can be
read from the file. For example, if the file contains three scans with names
``Scan 2``, ``Scan 4`` and ``Scan 6``:

.. code-block:: python

    >>> s2, s4, s6 = kp.load('patterns.h5', scans=[2, 4, 6])

Here the h5ebsd :func:`~kikuchipy.io.plugins.h5ebsd.file_reader` is called. If
only ``Scan 4`` is to be read, ``scans=4`` can be passed. The ``scans``
parameter is unnecessary if only ``Scan 2`` is to be read since reading the
first scan in the file is the default behaviour.

So far, only :ref:`saving patterns <save-patterns>` to KikuchiPy's own h5ebsd
format is supported. It is possible to write a new scan with a new scan number
to an existing, but closed, h5ebsd file in the KikuchiPy format, e.g. one
containing only ``Scan 1``, by passing:

.. code-block:: python

    >>> s.save('patterns.h5', add_scan=True, scan_number=2)

Here the h5ebsd :func:`~kikuchipy.io.plugins.h5ebsd.file_writer` is called.

.. _nordif-format:

NORDIF binary
-------------

Patterns acquired using NORDIF's acquisition software are stored in a binary
file usually named `Pattern.dat`. Scan information is stored in a separate text
file usually named `Setting.txt`, and both files usually reside in the same
directory. If this is the case, the patterns can be loaded by passing the file
name as the only parameter. If this is not the case, the setting file can be
passed upon loading:

.. code-block:: python

    >>> s = kp.load('Pattern.dat', setting_file='/somewhere/Setting_new.txt')

Here the NORDIF :func:`~kikuchipy.io.plugins.nordif.file_reader` is called. If
the scan information, i.e. scan and pattern size, in the setting file is
incorrect or the setting file is lost, patterns can be loaded by passing:

.. code-block:: python

    >>> s = kp.load('filename.dat', scan_size=(10, 20), pattern_size=(60, 60))

If a static background pattern named `Background acquisition.bmp` is stored in
the same directory as the pattern file, this is stored in ``metadata`` upon
loading.

Patterns can also be :ref:`saved to a NORDIF binary file <save-patterns>`, upon
which the NORDIF :func:`~kikuchipy.io.plugins.nordif.file_writer` is called.
Note, however, that so far no new setting file is created upon saving.

.. _from-kikuchipy-into-other-software:

From KikuchiPy into other software
==================================

Patterns saved in the :ref:`h5ebsd format <h5ebsd-format>` can be read by the
dictionary indexing and related routines in
`EMsoft <http://vbff.materials.cmu.edu/EMsoft>`_ using the `EMEBSD` reader.
Those routines in EMsoft also have a `NORDIF` reader.

Patterns saved in the :ref:`h5ebsd format <h5ebsd-format>` can of course be read
in Python like any other HDF5 data set:

.. code-block:: python

    >>> import h5py
    >>> with h5py.File('/path/to/patterns.h5', mode='r') as f:
            patterns = f['Scan 1/EBSD/Data/patterns'][()]

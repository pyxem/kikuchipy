=============
API reference
=============

This reference manual details the public classes, modules and functions in
kikuchipy, as generated from their docstrings. Many of the docstrings contain
examples, however, see the user guide for how to use kikuchipy.

.. caution::

    kikuchipy is in an alpha stage, and there will likely be breaking changes
    with each release.

Signals
=======

EBSD
----

All methods listed here are also available to
:class:`~kikuchipy.signals.LazyEBSD` objects.

.. currentmodule:: kikuchipy.signals.EBSD

.. autosummary::
    adaptive_histogram_equalization
    average_neighbour_patterns
    fft_filter
    get_decomposition_model
    get_dynamic_background
    get_image_quality
    get_virtual_bse_intensity
    normalize_intensity
    plot_virtual_bse_intensity
    rebin
    remove_dynamic_background
    remove_static_background
    rescale_intensity
    save
    set_detector_calibration
    set_experimental_parameters
    set_phase_parameters
    set_scan_calibration

.. autoclass:: kikuchipy.signals.EBSD
    :members:
    :undoc-members:
    :inherited-members: Signal2D
    :show-inheritance:

....

These methods are exclusive to LazyEBSD objects.

.. currentmodule:: kikuchipy.signals.LazyEBSD

.. autosummary::
    get_decomposition_model_write

.. autoclass:: kikuchipy.signals.LazyEBSD
    :members:
    :undoc-members:
    :show-inheritance:

....

EBSDMasterPattern
-----------------

All methods listed here are also available to
:class:`~kikuchipy.signals.LazyEBSDMasterPattern` objects.

.. currentmodule:: kikuchipy.signals.EBSDMasterPattern

.. autosummary::
    normalize_intensity
    rescale_intensity
    set_simulation_parameters
    set_phase_parameters

.. autoclass:: kikuchipy.signals.EBSDMasterPattern
    :members:
    :undoc-members:
    :inherited-members: Signal2D
    :show-inheritance:

....

There are no methods exclusive to LazyEBSDMasterPattern objects.

.. autoclass:: kikuchipy.signals.LazyEBSDMasterPattern
    :members:
    :undoc-members:
    :show-inheritance:

....

VirtualBSEImage
---------------

.. currentmodule:: kikuchipy.signals.VirtualBSEImage

.. autosummary::
    normalize_intensity
    rescale_intensity

.. autoclass:: kikuchipy.signals.VirtualBSEImage
    :members:
    :undoc-members:
    :inherited-members: Signal2D

....

Utilities
---------

.. currentmodule:: kikuchipy.signals.util

.. automodule:: kikuchipy.signals.util
    :members:
    :undoc-members:

....

Generators
==========

Generators producing signals. This workflow with generators is adopted from
`pyxem <http://github.com/pyxem/pyxem>`_.

VirtualBSEGenerator
-------------------

.. currentmodule:: kikuchipy.generators.VirtualBSEGenerator

.. autosummary::
    get_images_from_grid
    get_rgb_image
    plot_grid
    roi_from_grid

.. autoclass:: kikuchipy.generators.VirtualBSEGenerator
    :members:
    :undoc-members:

Utilities
---------

.. currentmodule:: kikuchipy.generators.util

.. automodule:: kikuchipy.generators.util
    :members:
    :undoc-members:

....

Pattern
=======

Single pattern processing
-------------------------

This module mainly includes functions operating on single EBSD patterns as
:class:`numpy.ndarray`.

.. currentmodule:: kikuchipy.pattern

.. autosummary::
    fft
    fft_filter
    fft_frequency_vectors
    fft_spectrum
    get_dynamic_background
    get_image_quality
    ifft
    normalize_intensity
    remove_dynamic_background
    rescale_intensity

.. automodule:: kikuchipy.pattern
    :members:
    :undoc-members:

....

Chunk processing
----------------

This module includes functions for operating on :class:`numpy.ndarray` or
:class:`dask.array.Array` chunks of EBSD patterns.

.. currentmodule:: kikuchipy.pattern.chunk

.. autosummary::
    adaptive_histogram_equalization
    average_neighbour_patterns
    fft_filter
    get_dynamic_background
    get_image_quality
    normalize_intensity
    remove_dynamic_background
    remove_static_background
    rescale_intensity

.. automodule:: kikuchipy.pattern.chunk
    :members:
    :undoc-members:

....

Correlate
---------

.. currentmodule:: kikuchipy.pattern.correlate

.. automodule:: kikuchipy.pattern.correlate
    :members:
    :undoc-members:

....

Filters
=======

.. currentmodule:: kikuchipy.filters

.. automodule:: kikuchipy.filters
    :members:
    :undoc-members:

....

Input/output
============

.. automodule:: kikuchipy.io._io
    :members:
    :undoc-members:

These plugin functions import patterns and parameters from file formats into
:class:`~kikuchipy.signals.EBSD` or
:class:`~kikuchipy.signals.EBSDMasterPattern` (or
:class:`~kikuchipy.signals.LazyEBSD` or
:class:`~kikuchipy.signals.LazyEBSDMasterPattern` if loading lazily) objects.

....

h5ebsd
------

.. automodule:: kikuchipy.io.plugins.h5ebsd
    :members:
    :undoc-members:

....

NORDIF
------

.. automodule:: kikuchipy.io.plugins.nordif
    :members:
    :undoc-members:

....

EMsoft EBSD master pattern HDF5
-------------------------------

.. automodule:: kikuchipy.io.plugins.emsoft_ebsd_master_pattern
    :members:
    :undoc-members:

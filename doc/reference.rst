===================
kikuchipy reference
===================

This reference manual details all methods for the EBSD and EBSDMasterPattern
classes, utility functions, and readers of EBSD patterns and EBSD master
patterns included in kikuchipy, as generated from their docstrings. For learning
how to use kikuchipy, see the user guide pages in the sidebar.

EBSD
====

All methods listed here are also available to
:class:`~kikuchipy.signals.ebsd.LazyEBSD` objects.

.. currentmodule:: kikuchipy.signals.ebsd.EBSD

.. autosummary::
    adaptive_histogram_equalization
    average_neighbour_patterns
    fft_filter
    get_decomposition_model
    get_dynamic_background
    get_image_quality
    get_virtual_image
    normalize_intensity
    rebin
    remove_dynamic_background
    remove_static_background
    rescale_intensity
    save
    set_detector_calibration
    set_experimental_parameters
    set_phase_parameters
    set_scan_calibration
    virtual_backscatter_electron_imaging

.. autoclass:: kikuchipy.signals.ebsd.EBSD
    :members:
    :undoc-members:
    :show-inheritance:

....

These methods are exclusive to LazyEBSD objects.

.. currentmodule:: kikuchipy.signals.ebsd.LazyEBSD

.. autosummary::
    get_decomposition_model_write

.. autoclass:: kikuchipy.signals.ebsd.LazyEBSD
    :members:
    :undoc-members:
    :show-inheritance:

....

EBSDMasterPattern
=================

All methods listed here are also available to
:class:`~kikuchipy.signals.ebsd_master_pattern.LazyEBSDMasterPattern` objects.

.. currentmodule:: kikuchipy.signals.ebsd_master_pattern.EBSDMasterPattern

.. autosummary::
    set_simulation_parameters
    set_phase_parameters

.. autoclass:: kikuchipy.signals.ebsd_master_pattern.EBSDMasterPattern
    :members:
    :undoc-members:
    :show-inheritance:

....

There are no methods exclusive to LazyEBSDMasterPattern objects.

.. currentmodule:: kikuchipy.signals.ebsd_master_pattern.LazyEBSDMasterPattern

.. autoclass:: kikuchipy.signals.ebsd_master_pattern.LazyEBSDMasterPattern
    :members:
    :undoc-members:
    :show-inheritance:

....

Utilities
=========

Single pattern processing
-------------------------

.. currentmodule:: kikuchipy.util.pattern

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

.. automodule:: kikuchipy.util.pattern
    :members:

....

Chunk processing
----------------

.. currentmodule:: kikuchipy.util.chunk

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

.. automodule:: kikuchipy.util.chunk
    :members:

....

Window
------

.. automodule:: kikuchipy.util.window
    :members:
    :undoc-members:

....

Pattern similarity
------------------

.. automodule:: kikuchipy.util.pattern_similarity
    :members:
    :undoc-members:

....

Input/output
------------

.. automodule:: kikuchipy.io._io
    :members:
    :undoc-members:

.. automodule:: kikuchipy.util.io
    :members:
    :undoc-members:

....

Input/output plugins
====================

These plugin functions import patterns and parameters from file formats into
:class:`~kikuchipy.signals.ebsd.EBSD` or
:class:`~kikuchipy.signals.ebsd_master_pattern.EBSDMasterPattern` (or
:class:`~kikuchipy.signals.ebsd.LazyEBSD` or
:class:`~kikuchipy.signals.ebsd_master_pattern.LazyEBSDMasterPattern` if loading
lazily) objects.

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

EMsoft EBSD master pattern HDF5
-------------------------------

.. automodule:: kikuchipy.io.plugins.emsoft_ebsd_master_pattern
    :members:
    :undoc-members:

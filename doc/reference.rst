=============
API reference
=============

This reference manual details the public modules, classes, and functions in
kikuchipy, as generated from their docstrings. Many of the docstrings contain
examples, however, see the user guide for how to use kikuchipy.

.. caution::

    kikuchipy is in an alpha stage, so there will be breaking changes with each
    release.

.. module:: kikuchipy

The list of top modules (and the load function):

.. autosummary::
    crystallography
    data
    detectors
    draw
    filters
    generators
    indexing
    io
    load
    pattern
    projections
    signals
    simulations

....

crystallography
===============

.. automodule:: kikuchipy.crystallography

.. currentmodule:: kikuchipy.crystallography

.. autosummary::
    get_direct_structure_matrix
    get_reciprocal_metric_tensor
    get_reciprocal_structure_matrix

.. autofunction:: get_direct_structure_matrix
.. autofunction:: get_reciprocal_metric_tensor
.. autofunction:: get_reciprocal_structure_matrix

....

data
====

.. currentmodule:: kikuchipy.data

.. autosummary::
    nickel_ebsd_small
    nickel_ebsd_large
    nickel_ebsd_master_pattern_small

.. automodule:: kikuchipy.data
    :members:
    :undoc-members:

....

detectors
=========

.. automodule:: kikuchipy.detectors

.. currentmodule:: kikuchipy.detectors.ebsd_detector

.. autosummary::
    EBSDDetector

EBSDDetector
------------

.. currentmodule:: kikuchipy.detectors.ebsd_detector.EBSDDetector

.. autosummary::
    plot

.. autoclass:: kikuchipy.detectors.ebsd_detector.EBSDDetector
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

....

draw
====

.. automodule:: kikuchipy.draw

.. currentmodule:: kikuchipy.draw

markers
-------

.. currentmodule:: kikuchipy.draw.markers

.. autosummary::
    get_line_segment_list
    get_point_list
    get_text_list

.. autofunction:: get_line_segment_list
.. autofunction:: get_point_list
.. autofunction:: get_text_list

colors
------

.. automodule:: kikuchipy.draw.colors

....

filters
=======

.. automodule:: kikuchipy.filters

.. currentmodule:: kikuchipy.filters.window

.. autosummary::
    distance_to_origin
    highpass_fft_filter
    lowpass_fft_filter
    modified_hann
    Window

.. autofunction:: distance_to_origin
.. autofunction:: highpass_fft_filter
.. autofunction:: lowpass_fft_filter
.. autofunction:: modified_hann

Window
------

.. currentmodule:: kikuchipy.filters.window.Window

.. autosummary::
    is_valid
    make_circular
    plot
    shape_compatible

.. autoclass:: kikuchipy.filters.window.Window
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

....

generators
==========

.. automodule:: kikuchipy.generators

.. currentmodule:: kikuchipy.generators

.. autosummary::
    EBSDSimulationGenerator
    VirtualBSEGenerator
    virtual_bse_generator.get_rgb_image
    virtual_bse_generator.normalize_image

EBSDSimulationGenerator
-----------------------

.. currentmodule:: kikuchipy.generators.EBSDSimulationGenerator

.. autosummary::
    geometrical_simulation

.. autoclass:: kikuchipy.generators.EBSDSimulationGenerator
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

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
    :show-inheritance:

    .. automethod:: __init__

Other functions
---------------

.. currentmodule:: kikuchipy.generators.virtual_bse_generator

.. autofunction:: get_rgb_image
.. autofunction:: normalize_image

....

indexing
========

.. automodule:: kikuchipy.indexing

.. currentmodule:: kikuchipy.indexing

.. autosummary::
    StaticPatternMatching
    orientation_similarity_map
    merge_crystal_maps
    similarity_metrics

.. autoclass:: StaticPatternMatching
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __call__

.. autofunction:: orientation_similarity_map
.. autofunction:: merge_crystal_maps

similarity_metrics
------------------

.. currentmodule:: kikuchipy.indexing.similarity_metrics

.. autosummary::
    make_similarity_metric
    MetricScope
    ncc
    ndp

.. automodule:: kikuchipy.indexing.similarity_metrics
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: ncc
.. autofunction:: ndp

....

io
===

.. automodule:: kikuchipy.io

.. currentmodule:: kikuchipy.io

.. autosummary::
    _io.load
    plugins

.. autofunction:: kikuchipy.io._io.load

plugins
-------

.. automodule:: kikuchipy.io.plugins

.. currentmodule:: kikuchipy.io.plugins

.. autosummary::
    h5ebsd
    nordif
    emsoft_ebsd
    emsoft_ebsd_master_pattern

The plugins import patterns and parameters from file formats into
:class:`~kikuchipy.signals.EBSD` or
:class:`~kikuchipy.signals.EBSDMasterPattern` (or
:class:`~kikuchipy.signals.LazyEBSD` or
:class:`~kikuchipy.signals.LazyEBSDMasterPattern` if loading lazily) objects.

h5ebsd
~~~~~~

.. automodule:: kikuchipy.io.plugins.h5ebsd
    :members:
    :undoc-members:
    :show-inheritance:

nordif
~~~~~~

.. automodule:: kikuchipy.io.plugins.nordif
    :members:
    :undoc-members:
    :show-inheritance:

emsoft_ebsd
~~~~~~~~~~~

.. automodule:: kikuchipy.io.plugins.emsoft_ebsd
    :members:
    :undoc-members:
    :show-inheritance:

emsoft_ebsd_master_pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: kikuchipy.io.plugins.emsoft_ebsd_master_pattern
    :members:
    :undoc-members:
    :show-inheritance:

....

pattern
=======

Single and chunk pattern processing used by signals.

.. currentmodule:: kikuchipy.pattern

.. autosummary::
    chunk
    correlate
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

Functions operating on single EBSD patterns as :class:`numpy.ndarray`.

.. automodule:: kikuchipy.pattern
    :members:
    :undoc-members:

chunk
-----

Functions for operating on :class:`numpy.ndarray` or :class:`dask.array.Array`
chunks of EBSD patterns.

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

correlate
---------

.. currentmodule:: kikuchipy.pattern.correlate

.. automodule:: kikuchipy.pattern.correlate
    :members:
    :undoc-members:

....

projections
===========

.. automodule:: kikuchipy.projections

.. currentmodule:: kikuchipy.projections

.. autosummary::
    ebsd_projections
    hesse_normal_form.HesseNormalForm
    spherical_projection.SphericalProjection

ebsd_projections
----------------

.. currentmodule:: kikuchipy.projections.ebsd_projections

.. autosummary::
    detector2direct_lattice
    detector2reciprocal_lattice
    detector2sample

.. automodule:: kikuchipy.projections.ebsd_projections
    :members:
    :undoc-members:

HesseNormalForm
-----------------

.. automodule:: kikuchipy.projections.hesse_normal_form

.. autoclass:: kikuchipy.projections.hesse_normal_form.HesseNormalForm
    :members:
    :undoc-members:
    :show-inheritance:

SphericalProjection
--------------------

.. automodule:: kikuchipy.projections.spherical_projection

.. autoclass:: kikuchipy.projections.spherical_projection.SphericalProjection
    :members:
    :undoc-members:
    :show-inheritance:

....

signals
=======

.. automodule:: kikuchipy.signals

.. currentmodule:: kikuchipy.signals

.. autosummary::
    EBSD
    EBSDMasterPattern
    VirtualBSEImage
    util

EBSD
----

All methods listed here are also available to
:class:`~kikuchipy.signals.LazyEBSD` objects.

.. currentmodule:: kikuchipy.signals.EBSD

.. autosummary::
    adaptive_histogram_equalization
    average_neighbour_patterns
    match_patterns
    fft_filter
    get_average_neighbour_dot_product_map
    get_decomposition_model
    get_dynamic_background
    get_image_quality
    get_neighbour_dot_product_matrices
    get_virtual_bse_intensity
    match_patterns
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

These methods are exclusive to LazyEBSD objects.

.. currentmodule:: kikuchipy.signals.LazyEBSD

.. autosummary::
    get_decomposition_model_write

.. autoclass:: kikuchipy.signals.LazyEBSD
    :members:
    :undoc-members:
    :show-inheritance:

EBSDMasterPattern
-----------------

All methods listed here are also available to
:class:`~kikuchipy.signals.LazyEBSDMasterPattern` objects.

.. currentmodule:: kikuchipy.signals.EBSDMasterPattern

.. autosummary::
    get_patterns
    normalize_intensity
    rescale_intensity

.. autoclass:: kikuchipy.signals.EBSDMasterPattern
    :members:
    :inherited-members: Signal2D
    :show-inheritance:

There are no methods exclusive to LazyEBSDMasterPattern objects.

.. autoclass:: kikuchipy.signals.LazyEBSDMasterPattern
    :members:
    :undoc-members:
    :show-inheritance:

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

util
----

.. currentmodule:: kikuchipy.signals.util

.. autosummary::
    ebsd_metadata
    get_chunking
    get_dask_array
    metadata_nodes

.. automodule:: kikuchipy.signals.util
    :members:
    :undoc-members:

....

simulations
===========

.. automodule:: kikuchipy.simulations

.. currentmodule:: kikuchipy.simulations

.. autosummary::
    GeometricalEBSDSimulation
    features

GeometricalEBSDSimulation
-------------------------

.. currentmodule:: kikuchipy.simulations.GeometricalEBSDSimulation

.. autosummary::
    as_markers
    bands_as_markers
    pc_as_markers
    zone_axes_as_markers
    zone_axes_labels_as_markers

.. autoclass:: kikuchipy.simulations.GeometricalEBSDSimulation
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

features
--------

.. automodule:: kikuchipy.simulations.features

.. currentmodule:: kikuchipy.simulations.features

.. autosummary::
    KikuchiBand
    ZoneAxis

KikuchiBand
~~~~~~~~~~~

.. autoclass:: kikuchipy.simulations.features.KikuchiBand
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: from_highest_hkl, from_min_dspacing, symmetrise, unique

    .. automethod:: __init__
    .. automethod:: __getitem__

ZoneAxis
~~~~~~~~

.. autoclass:: kikuchipy.simulations.features.ZoneAxis
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: from_highest_hkl, from_min_dspacing, symmetrise, unique

    .. automethod:: __init__
    .. automethod:: __getitem__

=============
API reference
=============

This reference manual details the public modules, classes, and functions in kikuchipy,
as generated from their docstrings. Many of the docstrings contain examples, however,
see the user guide for how to use kikuchipy.

.. caution::

    kikuchipy is in continuous development (alpha stage), so expect
    some breaking changes with each release.

.. module:: kikuchipy

The list of top modules and the load function:

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
    silicon_ebsd_moving_screen_in
    silicon_ebsd_moving_screen_out5mm
    silicon_ebsd_moving_screen_out10mm

.. automodule:: kikuchipy.data
    :members:
    :undoc-members:

....

detectors
=========

.. automodule:: kikuchipy.detectors
.. currentmodule:: kikuchipy.detectors

.. autosummary::
    EBSDDetector
    PCCalibrationMovingScreen

EBSDDetector
------------

.. currentmodule:: kikuchipy.detectors.EBSDDetector

.. autosummary::
    deepcopy
    pc_bruker
    pc_emsoft
    pc_oxford
    pc_tsl
    plot

.. autoclass:: kikuchipy.detectors.EBSDDetector
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

PCCalibrationMovingScreen
-------------------------

.. currentmodule:: kikuchipy.detectors.PCCalibrationMovingScreen

.. autosummary::
    make_lines
    plot

.. autoclass:: kikuchipy.detectors.PCCalibrationMovingScreen
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

....

draw
====

.. automodule:: kikuchipy.draw

.. autosummary::
    colors
    get_rgb_navigator
    markers

.. autofunction:: get_rgb_navigator

markers
-------

.. automodule:: kikuchipy.draw.markers

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
    :members:
    :undoc-members:

....

filters
=======

.. automodule:: kikuchipy.filters

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

.. currentmodule:: kikuchipy.filters.Window

.. autosummary::
    is_valid
    make_circular
    plot
    shape_compatible

.. autoclass:: kikuchipy.filters.Window
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
    compute_refine_orientation_results
    compute_refine_orientation_projection_center_results
    compute_refine_projection_center_results
    orientation_similarity_map
    merge_crystal_maps
    similarity_metrics

.. autofunction:: compute_refine_orientation_results
.. autofunction:: compute_refine_orientation_projection_center_results
.. autofunction:: compute_refine_projection_center_results
.. autofunction:: orientation_similarity_map
.. autofunction:: merge_crystal_maps

similarity_metrics
------------------

.. automodule:: kikuchipy.indexing.similarity_metrics

.. currentmodule:: kikuchipy.indexing.similarity_metrics

.. autosummary::
    SimilarityMetric
    NormalizedCrossCorrelationMetric
    NormalizedDotProductMetric

.. autoclass:: kikuchipy.indexing.similarity_metrics.SimilarityMetric
    :members:

    .. automethod:: __init__

.. autoclass:: kikuchipy.indexing.similarity_metrics.NormalizedCrossCorrelationMetric
    :members:
    :show-inheritance:

    .. automethod:: __call__

.. autoclass:: kikuchipy.indexing.similarity_metrics.NormalizedDotProductMetric
    :members:
    :show-inheritance:

    .. automethod:: __call__

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
    emsoft_ebsd
    emsoft_ebsd_master_pattern
    h5ebsd
    nordif
    nordif_calibration_patterns
    oxford_binary

The plugins import patterns and parameters from file formats into
:class:`~kikuchipy.signals.EBSD` or :class:`~kikuchipy.signals.EBSDMasterPattern` (or
:class:`~kikuchipy.signals.LazyEBSD` or
:class:`~kikuchipy.signals.LazyEBSDMasterPattern` if loading lazily) objects.

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

nordif_calibration_patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: kikuchipy.io.plugins.nordif_calibration_patterns
    :members:
    :undoc-members:
    :show-inheritance:


oxford_binary
~~~~~~~~~~~~~

.. automodule:: kikuchipy.io.plugins.oxford_binary
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

Functions operating on single EBSD patterns as :class:`~numpy.ndarray`.

.. automodule:: kikuchipy.pattern
    :members:
    :undoc-members:

chunk
-----

Functions for operating on :class:`~numpy.ndarray` or :class:`~dask.array.Array` chunks
of EBSD patterns.

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

All methods listed here are also available to :class:`~kikuchipy.signals.LazyEBSD`
instances.

See :class:`̃hyperspy._signals.signal2d.Signal2D` for methods inherited from HyperSpy.

.. currentmodule:: kikuchipy.signals.EBSD

.. autosummary::
    adaptive_histogram_equalization
    average_neighbour_patterns
    dictionary_indexing
    fft_filter
    get_average_neighbour_dot_product_map
    get_decomposition_model
    get_dynamic_background
    get_image_quality
    get_neighbour_dot_product_matrices
    get_virtual_bse_intensity
    normalize_intensity
    plot_virtual_bse_intensity
    refine_orientation
    refine_orientation_projection_center
    refine_projection_center
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

These methods are exclusive to LazyEBSD instances.

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
:class:`~kikuchipy.signals.LazyEBSDMasterPattern` instances.

See :class:`~hyperspy._signals.signal2d.Signal2D` for methods inherited from HyperSpy.

.. currentmodule:: kikuchipy.signals.EBSDMasterPattern

.. autosummary::
    get_patterns
    normalize_intensity
    plot_spherical
    rescale_intensity

.. autoclass:: kikuchipy.signals.EBSDMasterPattern
    :members:
    :inherited-members: Signal2D
    :show-inheritance:

There are no methods exclusive to LazyEBSDMasterPattern instances.

.. autoclass:: kikuchipy.signals.LazyEBSDMasterPattern
    :members:
    :undoc-members:
    :show-inheritance:

VirtualBSEImage
---------------

See :class:`~hyperspy._signals.signal2d.Signal2D` for methods inherited from HyperSpy.

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

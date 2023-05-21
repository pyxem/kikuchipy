=========
Changelog
=========

kikuchipy is a library for processing, simulating and indexing of electron backscatter
diffraction (EBSD) patterns in Python, built on the tools for multi-dimensional data
analysis provided by the HyperSpy library: https://kikuchipy.org.

All user facing changes to this project are documented in this file. The format is based
on `Keep a Changelog <https://keepachangelog.com/en/1.1.0>`__, and this project tries
its best to adhere to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`__.

List entries are sorted in descending chronological order. Contributors to each release
were listed in alphabetical order by first name until version 0.7.0.

0.8.5 (2023-05-21)
==================

Fixed
-----
- Not-indexed points in crystal maps are handled correctly when merging.
  (`#639 <https://github.com/pyxem/kikuchipy/pull/639>`_)

0.8.4 (2023-04-07)
==================

Fixed
-----
- Points considered not-indexed in a crystal map are maintained after EBSD refinement.
  (`#632 <https://github.com/pyxem/kikuchipy/pull/632>`_)

Changed
-------
- EBSD detector returned from combined EBSD and projection center (PC) refinement now
  has PC values equal to the number of indexed points, accounting for points not being
  in the data, navigation mask *and* points considered as not-indexed. This means that
  it might not have a 2D navigation shape, even though the returned crystal map has.
  (`#632 <https://github.com/pyxem/kikuchipy/pull/632>`_)

0.8.3 (2023-03-23)
==================

Changed
-------
- ``EBSD.hough_indexing()`` info message now informs that the given projection center is
  in Bruker's convention. (`#628 <https://github.com/pyxem/kikuchipy/pull/628>`_)

0.8.2 (2023-03-14)
==================

Changed
-------
- Set minimal version of orix to >= 0.11.1.
  (`#623 <https://github.com/pyxem/kikuchipy/pull/623>`_)

0.8.1 (2023-02-20)
==================

Fixed
-----
- Hough indexing with PyEBSDIndex of a lazy EBSD signal requires not only PyOpenCL to be
  installed, but also for PyOpenCL to be able to create a context.
  (`#615 <https://github.com/pyxem/kikuchipy/pull/615>`_)
- Missing progressbars for ``EBSD`` methods ``average_neighbour_patterns()`` and
  ``fft_filter()`` reintroduced. (`#615 <https://github.com/pyxem/kikuchipy/pull/615>`_)

0.8.0 (2023-02-11)
==================

Added
-----
- ``kikuchipy.imaging.VirtualBSEImager`` replaces the
  ``kikuchipy.generators.VirtualBSEGenerator`` class.
  (`#608 <https://github.com/pyxem/kikuchipy/pull/608>`_)
- Adaptive histogram equalization is available to all signals.
  (`#606 <https://github.com/pyxem/kikuchipy/pull/606>`_)
- Option to return a new signal (lazy or not) instead of operating inplace is added to
  many methods in all classes via ``inplace`` and ``lazy_output`` boolean parameters.
  (`#605 <https://github.com/pyxem/kikuchipy/pull/605>`_)
- Lazy version of the ``VirtualBSEImage`` class.
  (`#605 <https://github.com/pyxem/kikuchipy/pull/605>`_)
- Allow providing a color for simulator reflections when plotting with Matplotlib.
  (`#599 <https://github.com/pyxem/kikuchipy/pull/599>`_)
- Passing pseudo-symmetry operators to orientation and orientation/PC EBSD refinement
  methods in order to find the best match among pseudo-symmetric variants.
  (`#598 <https://github.com/pyxem/kikuchipy/pull/598>`_)
- Saving and loading of an ``EBSDDetector``.
  (`#595 <https://github.com/pyxem/kikuchipy/pull/595>`_)
- EBSD refinement methods now return the number of function evaluations.
  (`#593 <https://github.com/pyxem/kikuchipy/pull/593>`_)
- Which points in a crystal map to refine can be controlled by passing a navigation
  mask. (`#593 <https://github.com/pyxem/kikuchipy/pull/593>`_)
- Which points to consider when merging crystal maps can be controlled by passing
  navigation masks.  (`#593 <https://github.com/pyxem/kikuchipy/pull/593>`_)
- Which patterns to do dictionary indexing of can be controlled by passing a navigation
  mask. (`#593 <https://github.com/pyxem/kikuchipy/pull/593>`_)
- Downsampling of EBSD patterns which maintain the data type by also rescaling to the
  data type range. (`#592 <https://github.com/pyxem/kikuchipy/pull/592>`_)
- Method to get a PyEBSDIndex ``EBSDIndexer`` instance from an ``EBSDDetector``,
  convenient for either indexing with PyEBSDIndex or for use with kikuchipy.
  (`#590 <https://github.com/pyxem/kikuchipy/pull/590>`_)
- Convenience function to get a ``CrystalMap`` from a PyEBSDIndex Hough indexing result
  array. (`#590 <https://github.com/pyxem/kikuchipy/pull/590>`_)
- PyEBSDIndex as an optional dependency.
  (`#590 <https://github.com/pyxem/kikuchipy/pull/590>`_)
- Two tutorials showing projection center (PC) fitting and extrapolation to obtain a
  plane of PCs to index a full dataset.
  (`#588 <https://github.com/pyxem/kikuchipy/pull/588>`_)
- Tutorial showing sloppy projection center (PC)/orientation optimization landscape of
  the Ni dataset from Jackson et al. (2019), replicating the results from Pang et al.
  (2020). (`#588 <https://github.com/pyxem/kikuchipy/pull/588>`_)
- Method ``EBSDDetector.fit_pc()`` to fit a plane using a projective or affine
  transformation to projection centers following work by Winkelmann and co-workers
  (2020). (`#587 <https://github.com/pyxem/kikuchipy/pull/587>`_)
- Method ``EBSDDetector.extrapolate_pc()`` to return a new detector with a plane of
  projection centers (PCs) extrapolated from a mean PC calculated from one or more PCs
  following work by Singh et al. (2017).
  (`#587 <https://github.com/pyxem/kikuchipy/pull/587>`_)
- Methods ``EBSDDetector.estimate_xtilt()`` and ``EBSDDetector.estimate_xtilt_ztilt()``
  to estimate the tilts about the detector X and Z axes which bring the detector plane
  normal parallel to the sample plane normal, following work by Winkelmann and
  co-workers (2020). (`#587 <https://github.com/pyxem/kikuchipy/pull/587>`_)
- Method ``EBSDDetector.plot_pc()`` to plot projection centers (PCs) in maps, scatter
  plots or in 3D. (`#587 <https://github.com/pyxem/kikuchipy/pull/587>`_)
- Convenience function ``kikuchipy.draw.plot_pattern_positions_in_map()`` to plot
  positions of selected patterns (typically calibration patterns) in a 2D map.
  (`#587 <https://github.com/pyxem/kikuchipy/pull/587>`_)
- EBSD signal returned from NORDIF calibration pattern reader tries to add the following
  new info to the original metadata: Shapes of area and region of interest (ROI), offset
  of ROI, calibration pattern indices and area overview image. All shapes and
  coordinates are given both in units of area overview image pixels and scaled according
  to the pixels in the ROI (actual navigation shape).
  (`#586 <https://github.com/pyxem/kikuchipy/pull/586>`_)
- Method ``EBSD.extract_grid()`` to get a new signal from grid positions evenly spaced
  in navigation space. (`#585 <https://github.com/pyxem/kikuchipy/pull/585>`_)
- Utility function ``grid_indices()`` to extract a smaller 1D or 2D grid of indices from
  a larger grid. (`#585 <https://github.com/pyxem/kikuchipy/pull/585>`_)
- Seven EBSD master pattern files simulated with EMsoft are available from
  ``ebsd_master_pattern()`` via the data module for download to the local cache:
  aluminium, nickel, silicon, austenite, ferrite, a chi-phase in steel and a sigma-phase
  in steel. (`#584 <https://github.com/pyxem/kikuchipy/pull/584>`_,
  `#607 <https://github.com/pyxem/kikuchipy/pull/607>`_)
- Some experimental EBSD datasets are available for download to the local cache via the
  data module: (50, 50) patterns of (480, 480) pixels from an Si wafer via
  ``si_wafer()``, ten full Ni datasets of (149, 200) patterns of (60, 60) pixels via
  ``ni_gain(number)`` (number 1-10) (parts of number 1 are used in
  ``nickel_ebsd_small()/large()``) and the calibration patterns of the ten Ni datasets,
  ``ni_gain_calibration(number)``.
  (`#584 <https://github.com/pyxem/kikuchipy/pull/584>`_,
  `#593 <https://github.com/pyxem/kikuchipy/pull/593>`_,
  `#607 <https://github.com/pyxem/kikuchipy/pull/607>`_)
- When using the following HyperSpy ``Signal2D`` methods via the ``EBSD`` class, the
  class attributes ``xmap``, ``static_background`` and ``detector`` are handled
  correctly, which they were not before: ``inav``, ``isig``, ``crop()``,
  ``crop_image()``. If handling fails, the old behavior is retained. This handling is
  experimental. (`#578 <https://github.com/pyxem/kikuchipy/pull/578>`_)
- ``EBSDDetector.crop()`` method to get a new detector with its shape cropped, also
  updating the PC values accordingly.
  (`#578 <https://github.com/pyxem/kikuchipy/pull/578>`_)

Changed
-------
- Minimal version of orix set to >= 0.11 and of numba set to >= 0.55.
  (`#608 <https://github.com/pyxem/kikuchipy/pull/608>`_)
- Added warnings when trying to perform adaptive histogram equalization on a signal with
  data in floating type or when some of the data is NaN.
  (`#606 <https://github.com/pyxem/kikuchipy/pull/606>`_)
- Dask arrays returned from EBSD refinement methods has the number of function
  evaluations as the second element after the score.
  (`#593 <https://github.com/pyxem/kikuchipy/pull/593>`_)
- Stricter phase comparison in EBSD refinement. The phase in the crystal map points to
  refine must have the same name, space group, point group and structure (atoms and
  lattice) as the master pattern phase.
  (`#593 <https://github.com/pyxem/kikuchipy/pull/593>`_)
- Passing two crystal maps with identical phases when merging returns a map with one
  phase instead of two and does not raise a warning, as before.
  (`#593 <https://github.com/pyxem/kikuchipy/pull/593>`_)
- Exclude documentation and tests from source distribution.
  (`#588 <https://github.com/pyxem/kikuchipy/pull/588>`_)
- Minimal version of HyperSpy increased to >= 1.7.3.
  (`#585 <https://github.com/pyxem/kikuchipy/pull/585>`_)
- Minimal version of Numba increased to >= 0.52.
  (`#585 <https://github.com/pyxem/kikuchipy/pull/585>`_)
- When binning the navigation dimension(s) with ``EBSD.rebin()``, the class attributes
  ``xmap`` and ``static_background`` are set to ``None`` and ``detector.pc`` is set to
  ``[0.5, 0.5, 0.5]`` in the appropriate navigation shape. If the signal dimension(s)
  are binned, the ``static_background`` is binned similarly while the ``detector.shape``
  and ``detector.binning`` are updated. If this handling of attributes fails, the old
  behavior is retained. This handling is experimental.
  (`#578 <https://github.com/pyxem/kikuchipy/pull/578>`_)
- EBSD signal loaded with ``nickel_ebsd_small()`` and ``nickel_ebsd_large()`` now
  contain crystal maps with orientations and detectors with PC values found from Hough
  indexing with PyEBSDIndex followed by orientation and PC refinement.
  (`#578 <https://github.com/pyxem/kikuchipy/pull/578>`_,
  `#584 <https://github.com/pyxem/kikuchipy/pull/584>`_)
- Minimal version of Matplotlib is 3.5.0 when installing optional dependencies with
  ``pip install kikuchipy[viz]`` since PyVista requires this.
  (`#578 <https://github.com/pyxem/kikuchipy/pull/578>`_)

Deprecated
----------
- ``kikuchipy.generators.VirtualBSEGenerator`` class is deprecated and will be removed
  in version 0.9. Use ``kikuchipy.imaging.VirtualBSEImager`` instead.
  (`#608 <https://github.com/pyxem/kikuchipy/pull/608>`_)
- The data module functions ``silicon_ebsd_moving_screen_x()``, where "x" is "in",
  "out5mm" or "out10mm", are deprecated and will be removed in v0.9. Use
  ``si_ebsd_moving_screen(distance)`` instead, where ``distance`` is ``0`` (in), ``5``
  or ``10``. (`#607 <https://github.com/pyxem/kikuchipy/pull/607>`_)
- The PyPI selector ``viz`` is replaced by ``all``, which installs all optional
  dependencies. ``viz`` will be removed in version 0.9. Install optional dependencies
  manually or via ``pip install kikuchipy[all]``.
  (`#590 <https://github.com/pyxem/kikuchipy/pull/590>`_)
- ``projections`` module with classes ``GnomonicProjection``, ``HesseNormalForm``,
  ``LambertProjection`` and ``SphericalProjection``. These will be removed in version
  0.9.0, as they are unused internally. If you depend on this module, please open an
  issue at https://github.com/pyxem/kikuchipy/issues.
  (`#577 <https://github.com/pyxem/kikuchipy/pull/577>`_)

Removed
-------
- ``mask`` parameter in EBSD refinement methods; use ``signal_mask`` instead.
  (`#577 <https://github.com/pyxem/kikuchipy/pull/577>`_)
- ``ebsd_projections`` module. (`#577 <https://github.com/pyxem/kikuchipy/pull/577>`_)

Fixed
-----
- Default ``EBSD.detector.shape`` is now correct when a detector is not passed upon
  initialization. (`#603 <https://github.com/pyxem/kikuchipy/pull/603>`_)
- Oxford Instruments .ebsp files of version 4 can now be read.
  (`#602 <https://github.com/pyxem/kikuchipy/pull/602>`_)
- When loading EBSD patterns from H5OINA files, the detector tilt and binning are
  available in the returned signal's ``detector`` attribute.
  (`#600 <https://github.com/pyxem/kikuchipy/pull/600>`_)
- Range of (kinematical) intensities in ``KikuchiPatternSimulator.plot()`` maximizes the
  strongest reflectors (make black) instead of minimizing the weakest reflectors (make
  white), which was the previous behavior.
  (`#599 <https://github.com/pyxem/kikuchipy/pull/599>`_)
- Inversion of ``signal_mask`` in the normalized cross-correlation and normalized dot
  product metrics is now done internally, to be in line with the docstrings (does not
  affect the use of this parameter and ``metric="ncc"`` or ``metric="ndp"`` in
  ``EBSD.dictionary_indexing()``).
  (`#593 <https://github.com/pyxem/kikuchipy/pull/593>`_)
- ``EBSDDetector.pc_average`` no longer rounds the PC to three decimals.
  (`#586 <https://github.com/pyxem/kikuchipy/pull/586>`_)
- Microscope magnification is now read correctly from EDAX h5ebsd files.
  (`#586 <https://github.com/pyxem/kikuchipy/pull/586>`_)
- kikuchipy h5ebsd reader can read a signal with an EBSD detector with a PC array of
  different navigation shape than determined from the HDF5 file's navigation shape
  (e.g. ``Scan 1/EBSD/Header/n_columns`` and ``n_rows``).
  (`#578 <https://github.com/pyxem/kikuchipy/pull/578>`_)

0.7.0 (2022-10-29)
==================

Added
-----
- Signal mask passed to EBSD orientation and projection center refinement methods is now
  applied to the experimental pattern as well.
  (`#573 <https://github.com/pyxem/kikuchipy/pull/573>`_)
- Dependency ``imageio`` needed for reading EBSD patterns in image files.
  (`#570 <https://github.com/pyxem/kikuchipy/pull/570>`_)
- Reader of an ``EBSD`` signal from all images in a directory assuming they are of the
  same shape and data type. (`#570 <https://github.com/pyxem/kikuchipy/pull/570>`_)
- Reader of an ``EBSD`` signal from EDAX TSL's binary UP1/UP2 file formats.
  (`#569 <https://github.com/pyxem/kikuchipy/pull/569>`_)
- Ability to project simulate patterns from a master pattern using varying projection
  centers (PCs) in ``EBSDMasterPattern.get_patterns()``. An example is added to the
  method to show this. (`#567 <https://github.com/pyxem/kikuchipy/pull/567>`_)
- Allow not setting ``energy`` parameter in ``EBSDMasterPattern.get_patterns()``, upon
  which the highest energy available is used.
  (`#567 <https://github.com/pyxem/kikuchipy/pull/567>`_)
- Improved handling of custom attributes ``xmap``, ``detector`` and
  ``static_background`` in ``EBSD`` and ``hemisphere``, ``phase`` and ``projection`` in
  EBSD/ECP master pattern classes when calling inherited HyperSpy ``Signal2D`` methods
  ``as_lazy()``, ``change_dtype()``, ``compute()``, ``deepcopy()``,
  ``set_signal_type()`` and ``squeeze()``.
  (`#564 <https://github.com/pyxem/kikuchipy/pull/564>`_)
- Reader of an electron channelig pattern (ECP) master pattern from an EMsoft HDF5 file
  into an ``ECPMasterPattern`` signal.
  (`#564 <https://github.com/pyxem/kikuchipy/pull/564>`_)
- Reader of a transmission kikuchi diffraction (TKD) master pattern from an EMsoft HDF5
  file into an ``EBSDMasterPattern`` signal.
  (`#564 <https://github.com/pyxem/kikuchipy/pull/564>`_)
- ``ECPMasterPattern`` class. (`#564 <https://github.com/pyxem/kikuchipy/pull/564>`_)
- Some internal logging which can be controlled via ``kikuchipy.set_log_level()``.
  (`#564 <https://github.com/pyxem/kikuchipy/pull/564>`_)
- Reader of an ``EBSD`` signal from Oxford Instrument's h5ebsd format (H5OINA).
  (`#562 <https://github.com/pyxem/kikuchipy/pull/562>`_)
- Figures of reference frames of other software added to the documentation.
  (`#552 <https://github.com/pyxem/kikuchipy/pull/552>`_)
- Whether to show progressbars from most signal methods (except indexing and refinement)
  can be controlled by passing ``show_progressbar`` or by setting HyperSpy's
  ``hs.preferences.General.show_progressbar`` (see their docs for details).
  (`#550 <https://github.com/pyxem/kikuchipy/pull/550>`_)

Changed
-------
- Documentation theme from *Furo* to *PyData*, as the growing API reference is easier to
  navigate with the latter. (`#574 <https://github.com/pyxem/kikuchipy/pull/574>`_)
- Use Rodrigues-Frank vector components (Rx, Ry, Rz) instead of Euler angles in EBSD
  orientation and projection center refinement methods. This means that if refinement is
  not directly but a Dask array is returned from any of these methods, the data which
  previously contained Euler angles now contain these vector components. This change was
  done to speed up refinement. (`#573 <https://github.com/pyxem/kikuchipy/pull/573>`_)
- Most of the ``EBSD`` metadata structure is removed, in an effort to move all relevant
  data to the attributes ``xmap``, ``static_background``, and ``detector``.
  (`#562 <https://github.com/pyxem/kikuchipy/pull/562>`_)
- h5ebsd plugin split into one plugin for each h5ebsd format (kikuchipy, EDAX TSL, and
  Bruker Nano).
  (`#562 <https://github.com/pyxem/kikuchipy/pull/562>`_)
- ``EBSDDetector.plot()`` and ``PCCalibrationMovingScreen.plot()`` parameter
  ``return_fig_ax`` renamed to ``return_figure``.
  (`#552 <https://github.com/pyxem/kikuchipy/pull/552>`_)
- Import modules lazily using the specification in `PEP 562
  <https://peps.python.org/pep-0562/>`__.
  (`#551 <https://github.com/pyxem/kikuchipy/pull/551>`_)
- Minimal version of HyperSpy increased to >= 1.7.1.
  (`#550 <https://github.com/pyxem/kikuchipy/pull/550>`_)
- ``progressbar`` parameter to ``show_progressbar`` in ``kikuchipy.data`` functions
  which accepts a ``allow_download`` parameter. If not given, the value is retreived
  from HyperSpy's preferences. (`#550 <https://github.com/pyxem/kikuchipy/pull/550>`_)

Deprecated
----------
- ``mask`` parameter in EBSD orientation and projection center refinement is deprecated
  in favor of ``signal_mask``, and will be removed in version 0.8.0.
  (`#573 <https://github.com/pyxem/kikuchipy/pull/573>`_)
- ``projections.ebsd_projections`` module.
  (`#563 <https://github.com/pyxem/kikuchipy/pull/563>`_)

Removed
-------
- ``EBSDSimulationGenerator`` and ``GeometricalEBSDSimulation`` (use
  ``KikuchiPatternSimulator`` and ``GeometricalKikuchiPatternSimulation`` instead) and
  ``simulations.features`` module.
  (`#563 <https://github.com/pyxem/kikuchipy/pull/563>`_)
- ``crystallography`` module. (`#563 <https://github.com/pyxem/kikuchipy/pull/563>`_)
- Options ``"north"`` and ``"south"`` for property
  ``EBSDMasterPattern.hemisphere`` and in the parameter ``"hemisphere"`` in
  ``kikuchipy.data.nickel_ebsd_master_pattern_small()``; use ``"upper"`` and ``"lower"``
  instead. (`#563 <https://github.com/pyxem/kikuchipy/pull/563>`_)
- Functions ``remove_static_background()``, ``remove_dynamic_background()`` and
  ``get_image_quality()`` from ``chunk`` module.
  (`#563 <https://github.com/pyxem/kikuchipy/pull/563>`_)
- Parameter ``relative`` in ``EBSD.remove_static_background()``.
  (`#563 <https://github.com/pyxem/kikuchipy/pull/563>`_)
- Functions ``ebsd_metadata()`` and ``metadata_nodes()`` which have been deprecated
  since v0.5. (`#550 <https://github.com/pyxem/kikuchipy/pull/550>`_,
  `#562 <https://github.com/pyxem/kikuchipy/pull/562>`_)
- Print information emitted from ``EBSD`` methods like ``remove_static_background()`` is
  removed. (`#550 <https://github.com/pyxem/kikuchipy/pull/550>`_)

Fixed
-----
- ``detector`` attribute of ``EBSD`` signal returned from the NORDIF
  calibration pattern reader is now an ``EBSDDetector`` and not just a dictionary.
  (`#569 <https://github.com/pyxem/kikuchipy/pull/569>`_)
- Silence dask warning about splitting large chunks in ``EBSD.dictionary_indexing()``.
  Memory use can be controlled by rechunking the dictionary or setting the ``rechunk``
  or ``n_per_iteration`` parameters.
  (`#567 <https://github.com/pyxem/kikuchipy/pull/567>`_)

0.6.1 (2022-06-17)
==================

Contributors
------------
- Håkon Wiik Ånes

Fixed
-----
- Incorrect filtering of zone axes labels in geometrical simulations.
  (`#544 <https://github.com/pyxem/kikuchipy/pull/544>`_)

0.6.0 (2022-06-16)
==================

Contributors
------------
- Håkon Wiik Ånes

Added
-----
- ``EBSDMasterPattern.plot_spherical()`` for plotting a master pattern in the
  stereographic projection on the 3D sphere.
  (`#536 <https://github.com/pyxem/kikuchipy/pull/536>`_)
- Projection of master pattern in the stereographic projection to the square Lambert
  projection via ``EBSDMasterPattern.to_lambert()``.
  (`#536 <https://github.com/pyxem/kikuchipy/pull/536>`_)
- New package dependencies on ``pyvista`` for 3D plotting and on ``pythreejs`` for the
  docs are introduced. (`#536 <https://github.com/pyxem/kikuchipy/pull/536>`_)
- Reduce time and memory use of the following ``kikuchipy.signals.EBSD`` methods by
  using ``hyperspy.signal.BaseSignal.map()``: ``remove_static_background()``,
  ``remove_dynamic_background()`` and ``get_image_quality()``.
  (`#527 <https://github.com/pyxem/kikuchipy/pull/527>`_)
- ``progressbar`` parameter to functions downloading external datasets in the data
  module. (`#515 <https://github.com/pyxem/kikuchipy/pull/515>`_)
- Support for Python 3.10. (`#504 <https://github.com/pyxem/kikuchipy/pull/504>`_)
- ``EBSD.static_background`` property for easier access to the background pattern.
  (`#475 <https://github.com/pyxem/kikuchipy/pull/475>`_)

Changed
-------
- Valid ``EBSDMasterPattern.hemisphere`` values from ``"north"`` and ``"south"`` to
  ``"upper"`` and ``"lower"``, respectively, to be in line with `orix`.
  (`#537 <https://github.com/pyxem/kikuchipy/pull/537>`_)
- Increase minimal version of diffsims to 0.5.
  (`#537 <https://github.com/pyxem/kikuchipy/pull/537>`_)
- Chunking of EBSD signal navigation dimensions in
  ``EBSD.average_neighbour_patterns()`` to reduce memory use.
  (`#532 <https://github.com/pyxem/kikuchipy/pull/532>`_)
- Remove requirement that the crystal map used for EBSD refinement has identical step
  size(s) to the EBSD signal's navigation axes. This raised an error previously, but now
  only emits a warning. (`#531 <https://github.com/pyxem/kikuchipy/pull/531>`_)
- Increase minimal version of HyperSpy to 1.7.
  (`#527 <https://github.com/pyxem/kikuchipy/pull/527>`_)
- Increase minimal version of SciPy to 1.7.
  (`#504 <https://github.com/pyxem/kikuchipy/pull/504>`_)

Deprecated
----------
- The ``kikuchipy.simulations.GeometricalEBSDSimulation`` class is deprecated and will
  be removed in version 0.7. Obtain
  ``kikuchipy.simulations.GeometricalKikuchiPatternSimulation``  via
  ``kikuchipy.simulations.KikuchiPatternSimulator.on_detector()`` instead. The
  ``kikuchipy.simulations.features`` module is also deprecated and will be removed in
  version 0.7. Obtain Kikuchi line and zone axis detector/gnomonic coordinates of a
  simulation via ``lines_coordinates()`` and ``zone_axes_coordinates()`` instead.
  (`#537 <https://github.com/pyxem/kikuchipy/pull/537>`_)
- The ``kikuchipy.generators.EBSDSimulationGenerator`` class is deprecated and will be
  removed in version 0.7. Use the ``kikuchipy.simulations.KikuchiPatternSimulator``
  class instead. (`#537 <https://github.com/pyxem/kikuchipy/pull/537>`_)
- The ``kikuchipy.crystallography.matrices`` module is deprecated and will be removed in
  version 0.7, access the matrices via :class:`diffpy.structure.lattice.Lattice`
  attributes instead. (`#537 <https://github.com/pyxem/kikuchipy/pull/537>`_)
- The following functions for processing of pattern chunks in the
  ``kikuchipy.pattern.chunk`` module are deprecated and will be removed in version 0.7:
  ``get_image_quality()``, ``remove_dynamic_background()`` and
  ``remove_static_background()``. Use the ``EBSD`` class for processing of many
  patterns. (`#527 <https://github.com/pyxem/kikuchipy/pull/527>`_,
  `#533 <https://github.com/pyxem/kikuchipy/pull/533>`_  )

Removed
-------
- The ``relative`` parameter in ``kikuchipy.signals.EBSD.remove_static_background()``.
  The parameter is accepted but not used. Passing it after this release will result in
  an error. (`#527 <https://github.com/pyxem/kikuchipy/pull/527>`_)

Fixed
-----
- Plotting of geometrical simulation markers on rectangular patterns.
  (`#537 <https://github.com/pyxem/kikuchipy/pull/537>`_)
- Hopefully prevent EBSD refinement tests using random data to fail on Azure.
  (`#465 <https://github.com/pyxem/kikuchipy/pull/465>`_)

0.5.8 (2022-05-16)
==================

Contributors
------------
- Håkon Wiik Ånes

Changed
-------
- Minimal version of ``orix`` is increased to 0.9.
  (`#520 <https://github.com/pyxem/kikuchipy/pull/520>`_)

Fixed
-----
- Internal use of ``orix.vector.Vector3d`` following ``orix``' 0.9.0 release.
  (`#520 <https://github.com/pyxem/kikuchipy/pull/520>`_)

0.5.7 (2022-01-10)
==================

Contributors
------------
- Håkon Wiik Ånes

Fixed
-----
- EBSD orientation refinement on Windows producing garbage results due to unpredictable
  behaviour in Numba function which converts Euler triplet to quaternion.
  (`#495 <https://github.com/pyxem/kikuchipy/pull/495>`_)

0.5.6 (2022-01-02)
==================

Contributors
------------
- Håkon Wiik Ånes

Added
-----
- Convenience function `get_rgb_navigator()` to create an RGB signal from an RGB image.
  (`#491 <https://github.com/pyxem/kikuchipy/pull/491>`_)

Changed
-------
- Pattern matching notebook to include orientation maps from orix.
  (`#491 <https://github.com/pyxem/kikuchipy/pull/491>`_)

0.5.5 (2021-12-12)
==================

Contributors
------------
- Håkon Wiik Ånes
- Zhou Xu

Fixed
-----
- Not flipping rows and columns when saving non-square patterns to kikuchipy's h5ebsd
  format. (`#486 <https://github.com/pyxem/kikuchipy/pull/486>`_)

0.5.4 (2021-11-17)
==================

Contributors
------------
- Håkon Wiik Ånes

Added
-----
- Optional parameters `rechunk` and `chunk_kwargs` to EBSD refinement methods to better
  control possible rechunking of pattern array before refinement.
  (`#470 <https://github.com/pyxem/kikuchipy/pull/470>`_)

Changed
-------
- When EBSD refinement methods don't immediately compute, they return a dask array
  instead of a list of delayed instances.
  (`#470 <https://github.com/pyxem/kikuchipy/pull/470>`_)

Fixed
-----
- Memory issue in EBSD refinement due to naive use of dask.delayed. Uses map_blocks()
  instead. (`#470 <https://github.com/pyxem/kikuchipy/pull/470>`_)

0.5.3 (2021-11-02)
==================

Contributors
------------
- Håkon Wiik Ånes
- Zhou Xu

Added
-----
- Printing of speed (patterns per second) of dictionary indexing and refinement.
  (`#461 <https://github.com/pyxem/kikuchipy/pull/461>`_)
- Restricted newest version of hyperspy>=1.6.5 due to incompatibility with h5py>=3.5.
  (`#461 <https://github.com/pyxem/kikuchipy/pull/461>`_)

Fixed
-----
- Handling of projection centers (PCs): Correct conversion from/to EMsoft's convention
  requires binning factor *and* detector pixel size. Conversion between TSL/Oxford and
  Bruker conventions correctly uses detector aspect ratio.
  (`#455 <https://github.com/pyxem/kikuchipy/pull/455>`_)

0.5.2 (2021-09-11)
==================

Contributors
------------
- Håkon Wiik Ånes

Changed
-------
- Add gnomonic circles as patches in axes returned from EBSDDetector.plot().
  (`#445 <https://github.com/pyxem/kikuchipy/pull/445>`_)
- Restrict lowest supported version of orix to >= 0.7.
  (`#444 <https://github.com/pyxem/kikuchipy/pull/444>`_)

0.5.1 (2021-09-01)
==================

Contributors
------------
- Håkon Wiik Ånes

Added
-----
- Automatic creation of a release using GitHub Actions, which will simplify and lead to
  more frequent patch releases. (`#433 <https://github.com/pyxem/kikuchipy/pull/433>`_)

0.5.0 (2021-08-31)
==================

Contributors
------------
- Eric Prestat
- Håkon Wiik Ånes
- Lars Andreas Hastad Lervik

Added
-----
- Possibility to specify whether to rechunk experimental and simulated data sets and
  which data type to use for dictionary indexing.
  (`#419 <https://github.com/pyxem/kikuchipy/pull/419>`_)
- How to use the new orientation and/or projection center refinements to the pattern
  matching notebook. (`#405 <https://github.com/pyxem/kikuchipy/pull/405>`_)
- Notebooks to the documentation as shorter or longer "Examples" that don't fit in the
  user guide. (`#403 <https://github.com/pyxem/kikuchipy/pull/403>`_)
- Refinement module for EBSD refinement. Allows for the refinement of
  orientations and/or projection center estimates.
  (`#387 <https://github.com/pyxem/kikuchipy/pull/387>`_)

Changed
-------
- If a custom metric is to be used for dictionary indexing, it must now be a class
  inheriting from an abstract *SimilarityMetric* class. This replaces the previous
  *SimilarityMetric* class and the *make_similarity_metric()* function.
  (`#419 <https://github.com/pyxem/kikuchipy/pull/419>`_)
- Dictionary indexing parameter *n_slices* to *n_per_iteration*.
  (`#419 <https://github.com/pyxem/kikuchipy/pull/419>`_)
- *merge_crystal_maps* parameter *metric* to *greater_is_better*.
  (`#419 <https://github.com/pyxem/kikuchipy/pull/419>`_)
- *orientation_similarity_map* parameter *normalized* is by default False.
  (`#419 <https://github.com/pyxem/kikuchipy/pull/419>`_)
- Dependency versions for dask >= 2021.8.1, fixing some memory issues encountered after
  2021.3.1, and HyperSpy >= 1.6.4. Remove importlib_metadata from package dependencies.
  (`#418 <https://github.com/pyxem/kikuchipy/pull/418>`_)
- Performance improvements to EBSD dictionary generation, giving a substantial speed-up.
  (`#405 <https://github.com/pyxem/kikuchipy/pull/405>`_)
- Rename projection methods from `project()`/`iproject()` to
  `vector2xy()`/`xy2vector()`. (`#405 <https://github.com/pyxem/kikuchipy/pull/405>`_)
- URLs of user guide topics have an extra "/user_guide/<topic>" added to them.
  (`#403 <https://github.com/pyxem/kikuchipy/pull/403>`_)

Deprecated
----------
- Custom EBSD metadata, meaning the *Acquisition_instrument.SEM.EBSD.Detector* and
  *Sample.Phases* nodes, as well as the EBSD *set_experimental_parameters()* and
  *set_phase_parameters()* methods. This will be removed in v0.6 The *static_background*
  metadata array will become available as an EBSD property.
  (`#428 <https://github.com/pyxem/kikuchipy/pull/428>`_)

Removed
-------
- *make_similarity_metric()* function is replaced by the need to create a class inheriting
  from a new abstract *SimilarityMetric* class, which provides more freedom over
  preparations of arrays before dictionary indexing.
  (`#419 <https://github.com/pyxem/kikuchipy/pull/419>`_)
- *EBSD.match_patterns()* is removed, use *EBSD.dictionary_indexing()* instead.
  (`#419 <https://github.com/pyxem/kikuchipy/pull/419>`_)
- kikuchipy.pattern.correlate module.
  (`#419 <https://github.com/pyxem/kikuchipy/pull/419>`_)

Fixed
-----
- Allow static background in EBSD metadata to be a Dask array.
  (`#413 <https://github.com/pyxem/kikuchipy/pull/413>`_)
- Set newest supported version of Sphinx to 4.0.2 so that nbsphinx works.
  (`#403 <https://github.com/pyxem/kikuchipy/pull/403>`_)

0.4.0 (2021-07-08)
==================

Contributors
------------
- Håkon Wiik Ånes

Added
-----
- Sample tilt about RD can be passed as part of an EBSDDetector. This can be used when
  projecting parts of master patterns onto a detector.
  (`#381 <https://github.com/pyxem/kikuchipy/pull/381>`_)
- Reader for uncompressed EBSD patterns stored in Oxford Instrument's binary .ebsp file
  format. (`#371 <https://github.com/pyxem/kikuchipy/pull/371>`_,
  `#391 <https://github.com/pyxem/kikuchipy/pull/391>`_)
- Unit testing of docstring examples.
  (`#350 <https://github.com/pyxem/kikuchipy/pull/350>`_)
- Support for Python 3.9. (`#348 <https://github.com/pyxem/kikuchipy/pull/348>`_)
- Projection/pattern center calibration via the moving screen technique in a
  kikuchipy.detectors.calibration module.
  (`#322 <https://github.com/pyxem/kikuchipy/pull/322>`_)
- Three single crystal Si EBSD patterns, from the same sample position but with varying
  detector distances, to the data module (via external repo).
  (`#320 <https://github.com/pyxem/kikuchipy/pull/320>`_)
- Reading of NORDIF calibration patterns specified in a setting file into an EBSD
  signal. (`#317 <https://github.com/pyxem/kikuchipy/pull/317>`_)

Changed
-------
- Only return figure from kikuchipy.filters.Window.plot() if desired, also add a
  colorbar only if desired. (`#375 <https://github.com/pyxem/kikuchipy/pull/375>`_)

Deprecated
----------
- The kikuchipy.pattern.correlate module will be removed in v0.5. Use
  kikuchipy.indexing.similarity_metrics instead.
  (`#377 <https://github.com/pyxem/kikuchipy/pull/377>`_)
- Rename the EBSD.match_patterns() method to EBSD.dictionary_indexing().
  match_patterns() will be removed in v0.5.
  (`#376 <https://github.com/pyxem/kikuchipy/pull/376>`_)

Fixed
-----
- Set minimal requirement of importlib_metadata to v3.6 so Binder can run user guide
  notebooks with HyperSpy 1.6.3. (`#395 <https://github.com/pyxem/kikuchipy/pull/395>`_)
- Row (y) coordinate array returned with the crystal map from dictionary indexing is
  correctly sorted. (`#392 <https://github.com/pyxem/kikuchipy/pull/392>`_)
- Deep copying EBSD and EBSDMasterPattern signals carry over, respectively, `xmap` and
  `detector`, and `phase`, `hemisphere` and `projection` properties
  (`#356 <https://github.com/pyxem/kikuchipy/pull/356>`_).
- Scaling of region of interest coordinates used in virtual backscatter electron imaging
  to physical coordinates. (`#349 <https://github.com/pyxem/kikuchipy/pull/349>`_)

0.3.4 (2021-05-26)
==================

Contributors
------------
- Håkon Wiik Ånes

Added
-----
- Restricted newest version of dask<=2021.03.1 and pinned orix==0.6.0.
  (`#360 <https://github.com/pyxem/kikuchipy/pull/360>`_)

0.3.3 (2021-04-18)
==================

Contributors
------------
- Håkon Wiik Ånes
- Ole Natlandsmyr

Fixed
-----
- Reading of EBSD patterns from Bruker h5ebsd with a region of interest.
  (`#339 <https://github.com/pyxem/kikuchipy/pull/339>`_)
- Merging of (typically refined) crystal maps, where either a simulation indices array
  is not present or the array contains more indices per point than scores.
  (`#335 <https://github.com/pyxem/kikuchipy/pull/335>`_)
- Bugs in getting plot markers from geometrical EBSD simulation.
  (`#334 <https://github.com/pyxem/kikuchipy/pull/334>`_)
- Passing a static background pattern to EBSD.remove_static_background() for a
  non-square detector dataset works.
  (`#331 <https://github.com/pyxem/kikuchipy/pull/331>`_)

0.3.2 (2021-02-01)
==================

Contributors
------------
- Håkon Wiik Ånes

Fixed
-----
- Deletion of temporary files saved to temporary directories in user guide.
  (`#312 <https://github.com/pyxem/kikuchipy/pull/312>`_)
- Pattern matching sometimes failing to generate a crystal map due to incorrect creation
  of spatial arrays. (`#307 <https://github.com/pyxem/kikuchipy/pull/307>`_)

0.3.1 (2021-01-22)
==================

Contributors
------------
- Håkon Wiik Ånes

Fixed
-----
- Version link Binder uses to make the Jupyter Notebooks run in the browser.
  (`#301 <https://github.com/pyxem/kikuchipy/pull/301>`_)

0.3.0 (2021-01-22)
==================

Details of all development associated with this release is listed below and in `this
GitHub milestone <https://github.com/pyxem/kikuchipy/milestone/3?closed=1>`_.

Contributors
------------
- Håkon Wiik Ånes
- Lars Andreas Hastad Lervik
- Ole Natlandsmyr

Added
-----
- Calculation of an average dot product map, or just the dot product matrices.
  (`#280 <https://github.com/pyxem/kikuchipy/pull/280>`_)
- A nice gallery to the documentation with links to each user guide page.
  (`#285 <https://github.com/pyxem/kikuchipy/pull/285>`_)
- Support for writing/reading an EBSD signal with 1 or 0 navigation axes to/from a
  kikuchipy h5ebsd file.
  (`#276 <https://github.com/pyxem/kikuchipy/pull/276>`_)
- Better control over dask array chunking when processing patterns.
  (`#275 <https://github.com/pyxem/kikuchipy/pull/275>`_)
- User guide notebook showing basic pattern matching.
  (`#263 <https://github.com/pyxem/kikuchipy/pull/263>`_)
- EBSD.detector property storing an EBSDDetector.
  (`#262 <https://github.com/pyxem/kikuchipy/pull/262>`_)
- Link to Binder in README and in the notebooks for running them in the browser.
  (`#257 <https://github.com/pyxem/kikuchipy/pull/257>`_)
- Creation of dictionary of dynamically simulated EBSD patterns from a master pattern in
  the square Lambert projection. (`#239 <https://github.com/pyxem/kikuchipy/pull/239>`_)
- A data module with a small Nickel EBSD data set and master pattern, and a larger EBSD
  data set downloadable via the module. Two dependencies, pooch and tqdm, are added
  along with this module.
  (`#236 <https://github.com/pyxem/kikuchipy/pull/236>`_,
  `#237 <https://github.com/pyxem/kikuchipy/pull/237>`_,
  `#243 <https://github.com/pyxem/kikuchipy/pull/243>`_)
- Pattern matching of EBSD patterns with a dictionary of pre-computed simulated patterns
  with known crystal orientations, and related useful tools
  (`#231 <https://github.com/pyxem/kikuchipy/pull/231>`_,
  `#233 <https://github.com/pyxem/kikuchipy/pull/233>`_,
  `#234 <https://github.com/pyxem/kikuchipy/pull/234>`_): (1) A framework for creation
  of similarity metrics used in pattern matching, (2) computation of an orientation
  similarity map from indexing results, and (3) creation of a multi phase crystal map
  from single phase maps from pattern matching.
- EBSD.xmap property storing an orix CrystalMap.
  (`#226 <https://github.com/pyxem/kikuchipy/pull/226>`_)
- Dependency on the diffsims package for handling of electron scattering and
  diffraction. (`#220 <https://github.com/pyxem/kikuchipy/pull/220>`_)
- Square Lambert mapping, and its inverse, from points on the unit sphere to a 2D square
  grid, as implemented in Callahan and De Graef (2013).
  (`#214 <https://github.com/pyxem/kikuchipy/pull/214>`_)
- Geometrical EBSD simulations, projecting a set of Kikuchi bands and zone axes onto a
  detector, which can be added to an EBSD signal as markers.
  (`#204 <https://github.com/pyxem/kikuchipy/pull/204>`_,
  `#219 <https://github.com/pyxem/kikuchipy/pull/219>`_,
  `#232 <https://github.com/pyxem/kikuchipy/pull/232>`_)
- EBSD detector class to handle detector parameters, including detector pixels' gnomonic
  coordinates. EBSD reference frame documentation.
  (`#204 <https://github.com/pyxem/kikuchipy/pull/204>`_,
  `#215 <https://github.com/pyxem/kikuchipy/pull/215>`_)
- Reader for EMsoft's simulated EBSD patterns returned by their EMEBSD.f90 program.
  (`#202 <https://github.com/pyxem/kikuchipy/pull/202>`_)

Changed
-------
- The feature maps notebook to include how to obtain an average dot product map and dot
  product matrices for an EBSD signal.
  (`#280 <https://github.com/pyxem/kikuchipy/pull/280>`_)
- Averaging EBSD patterns with nearest neighbours now rescales to input data type range,
  thus loosing relative intensities, to avoid clipping intensities.
  (`#280 <https://github.com/pyxem/kikuchipy/pull/280>`_)
- Dependency requirement of diffsims from >= 0.3 to >= 0.4
  (`#282 <https://github.com/pyxem/kikuchipy/pull/282>`_)
- Name of hemisphere axis in EBSDMasterPattern from "y" to "hemisphere".
  (`#275 <https://github.com/pyxem/kikuchipy/pull/275>`_)
- Replace Travis CI with GitHub Actions.
  (`#250 <https://github.com/pyxem/kikuchipy/pull/250>`_)
- The EBSDMasterPattern gets phase, hemisphere and projection properties.
  (`#246 <https://github.com/pyxem/kikuchipy/pull/246>`_)
- EMsoft EBSD master pattern plugin can read a single energy pattern. Parameter
  `energy_range` changed to `energy`.
  (`240 <https://github.com/pyxem/kikuchipy/pull/240>`_)
- Migrate user guide from reST files to Jupyter Notebooks converted to HTML with the
  `nbsphinx` package.
  (`#236 <https://github.com/pyxem/kikuchipy/pull/236>`_,
  `#237 <https://github.com/pyxem/kikuchipy/pull/237>`_,
  `#244 <https://github.com/pyxem/kikuchipy/pull/244>`_,
  `#245 <https://github.com/pyxem/kikuchipy/pull/245>`_,
  `#279 <https://github.com/pyxem/kikuchipy/pull/279>`_,
  `#245 <https://github.com/pyxem/kikuchipy/pull/245>`_,
  `#279 <https://github.com/pyxem/kikuchipy/pull/279>`_,
  `#281 <https://github.com/pyxem/kikuchipy/pull/281>`_)
- Move GitHub repository to the pyxem organization. Update relevant URLs.
  (`#198 <https://github.com/pyxem/kikuchipy/pull/198>`_)
- Allow scikit-image >= 0.16. (`#196 <https://github.com/pyxem/kikuchipy/pull/196>`_)
- Remove language_version in pre-commit config file.
  (`#195 <https://github.com/pyxem/kikuchipy/pull/195>`_)

Removed
-------
- The EBSDMasterPattern and EBSD metadata node Sample.Phases, to be replaced
  by class attributes. The set_phase_parameters() method is removed from both
  classes, and the set_simulation_parameters() is removed from the former class.
  (`#246 <https://github.com/pyxem/kikuchipy/pull/246>`_)

Fixed
-----
- IndexError in neighbour pattern averaging
  (`#280 <https://github.com/pyxem/kikuchipy/pull/280>`_)
- Reading of square Lambert projections from EMsoft's master pattern file now sums
  contributions from asymmetric positions correctly.
  (`#255 <https://github.com/pyxem/kikuchipy/pull/255>`_)
- NumPy array creation when calculating window pixel's distance to the origin is not
  ragged anymore. (`#221 <https://github.com/pyxem/kikuchipy/pull/221>`_)

0.2.2 (2020-05-24)
==================

This is a patch release that fixes reading of EBSD data sets from h5ebsd files with
arbitrary scan group names.

Contributors
------------
- Håkon Wiik Ånes

Fixed
-------
- Allow reading of EBSD patterns from h5ebsd files with arbitrary scan group names, not
  just "Scan 1", "Scan 2", etc., like was the case before.
  (`#188 <https://github.com/pyxem/kikuchipy/pull/188>`_)

0.2.1 (2020-05-20)
==================

This is a patch release that enables installing kikuchipy 0.2 from Anaconda and not just
PyPI.

Contributors
------------
- Håkon Wiik Ånes

Changed
-------
- Use numpy.fft instead of scipy.fft because HyperSpy requires scipy < 1.4 on
  conda-forge, while scipy.fft was introduced in scipy 1.4.
  (`#180 <https://github.com/pyxem/kikuchipy/pull/180>`_)

Fixed
-----
- With the change above, kikuchipy 0.2 should be installable from Anaconda and not just
  PyPI. (`#180 <https://github.com/pyxem/kikuchipy/pull/180>`_)

0.2.0 (2020-05-19)
==================

Details of all development associated with this release are available `here
<https://github.com/pyxem/kikuchipy/milestone/2?closed=1>`_.

Contributors
------------
- Håkon Wiik Ånes
- Tina Bergh

Added
-----
- Jupyter Notebooks with tutorials and example workflows available.
- Grey scale and RGB virtual backscatter electron (BSE) images can be easily generated
  with the VirtualBSEGenerator class. The generator return objects of the new signal
  class VirtualBSEImage, which inherit functionality from HyperSpy's Signal2D class.
  (`#170 <https://github.com/pyxem/kikuchipy/pull/170>`_)
- EBSD master pattern class and reader of master patterns from EMsoft's EBSD master
  pattern file. (`#159 <https://github.com/pyxem/kikuchipy/pull/159>`_)
- Python 3.8 support. (`#157 <https://github.com/pyxem/kikuchipy/pull/157>`_)
- The public API has been restructured. The pattern processing used by the EBSD class is
  available in the kikuchipy.pattern subpackage, and filters/kernels used in frequency
  domain filtering and pattern averaging are available in the kikuchipy.filters
  subpackage.
  (`#169 <https://github.com/pyxem/kikuchipy/pull/169>`_)
- Intensity normalization of scan or single patterns.
  (`#157 <https://github.com/pyxem/kikuchipy/pull/157>`_)
- Fast Fourier Transform (FFT) filtering of scan or single patterns using SciPy's fft
  routines and `Connelly Barnes' filterfft
  <https://www.connellybarnes.com/code/python/filterfft>`_.
  (`#157 <https://github.com/pyxem/kikuchipy/pull/157>`_)
- Numba dependency to improve pattern rescaling and normalization.
  (`#157 <https://github.com/pyxem/kikuchipy/pull/157>`_)
- Computing of the dynamic background in the spatial or frequency domain for scan or
  single patterns. (`#157 <https://github.com/pyxem/kikuchipy/pull/157>`_)
- Image quality (IQ) computation for scan or single patterns based on N. C. K. Lassen's
  definition. (`#157 <https://github.com/pyxem/kikuchipy/pull/157>`_)
- Averaging of patterns with nearest neighbours with an arbitrary kernel, e.g.
  rectangular or Gaussian. (`#134 <https://github.com/pyxem/kikuchipy/pull/134>`_)
- Window/kernel/filter/mask class to handle such things, e.g. for pattern averaging or
  filtering in the frequency or spatial domain. Available in the kikuchipy.filters
  module.
  (`#134 <https://github.com/pyxem/kikuchipy/pull/134>`_,
  `#157 <https://github.com/pyxem/kikuchipy/pull/157>`_)

Changed
-------
- Renamed five EBSD methods: static_background_correction to remove_static_background,
  dynamic_background_correction to remove_dynamic_background, rescale_intensities to
  rescale_intensity, virtual_backscatter_electron_imaging to plot_virtual_bse_intensity,
  and get_virtual_image to get_virtual_bse_intensity.
  (`#157 <https://github.com/pyxem/kikuchipy/pull/157>`_,
  `#170 <https://github.com/pyxem/kikuchipy/pull/170>`_)
- Renamed kikuchipy_metadata to ebsd_metadata.
  (`#169 <https://github.com/pyxem/kikuchipy/pull/169>`_)
- Source code link in the documentation should point to proper GitHub line. This
  `linkcode_resolve` in the `conf.py` file is taken from SciPy.
  (`#157 <https://github.com/pyxem/kikuchipy/pull/157>`_)
- Read the Docs CSS style. (`#157 <https://github.com/pyxem/kikuchipy/pull/157>`_)
- New logo with a gradient from experimental to simulated pattern (with EMsoft), with a
  color gradient from the plasma color maps.
  (`#157 <https://github.com/pyxem/kikuchipy/pull/157>`_)
- Dynamic background correction can be done faster due to Gaussian blurring in the
  frequency domain to get the dynamic background to remove.
  (`#157 <https://github.com/pyxem/kikuchipy/pull/157>`_)

Removed
-------
- Explicit dependency on scikit-learn (it is imported via HyperSpy).
  (`#168 <https://github.com/pyxem/kikuchipy/pull/168>`_)
- Dependency on pyxem. Parts of their virtual imaging methods are adapted here---a big
  thank you to the pyxem/HyperSpy team!
  (`#168 <https://github.com/pyxem/kikuchipy/pull/168>`_)

Fixed
-----
- RtD builds documentation with Python 3.8 (fixed problem of missing .egg leading build
  to fail). (`#158 <https://github.com/pyxem/kikuchipy/pull/158>`_)

0.1.3 (2020-05-11)
==================

kikuchipy is an open-source Python library for processing and analysis of electron
backscatter diffraction patterns: https://kikuchipy.org.

This is a patch release. It is anticipated to be the final release in the `0.1.x`
series.

Added
-----
- Package installation with Anaconda via the `conda-forge channel
  <https://anaconda.org/conda-forge/kikuchipy/>`_.

Fixed
-----
- Static and dynamic background corrections are done at float 32-bit precision, and not
  integer 16-bit.
- Chunking of static background pattern.
- Chunking of patterns in the h5ebsd reader.

0.1.2 (2020-01-09)
==================

kikuchipy is an open-source Python library for processing and analysis of electron
backscatter diffraction patterns: https://kikuchipy.org.

This is a bug-fix release that ensures, unlike the previous bug-fix release, that
necessary files are downloaded when installing from PyPI.

0.1.1 (2020-01-04)
==================

This is a bug fix release that ensures that necessary files are uploaded to PyPI.

0.1.0 (2020-01-04)
==================

We're happy to announce the release of kikuchipy v0.1.0!

kikuchipy is an open-source Python library for processing and analysis of electron
backscatter diffraction (EBSD) patterns. The library builds upon the tools for
multi-dimensional data analysis provided by the HyperSpy library.

For more information, a user guide, and the full reference API documentation, please
visit: https://kikuchipy.org.

This is the initial pre-release, where things start to get serious... seriously fun!

Features
--------
- Load EBSD patterns and metadata from the NORDIF binary format (.dat), or Bruker Nano's
  or EDAX TSL's h5ebsd formats (.h5) into an ``EBSD`` object, e.g. ``s``, based upon
  HyperSpy's `Signal2D` class, using ``s = kp.load()``. This ensures easy access to
  patterns and metadata in the attributes ``s.data`` and ``s.metadata``, respectively.
- Save EBSD patterns to the NORDIF binary format (.dat) and our own h5ebsd format (.h5),
  using ``s.save()``. Both formats are readable by EMsoft's NORDIF and EMEBSD readers,
  respectively.
- All functionality in kikuchipy can be performed both directly and lazily (except some
  multivariate analysis algorithms). The latter means that all operations on a scan,
  including plotting, can be done by loading only necessary parts of the scan into
  memory at a time. Ultimately, this lets us operate on scans larger than memory using
  all of our cores.
- Visualize patterns easily with HyperSpy's powerful and versatile ``s.plot()``. Any
  image of the same navigation size, e.g. a virtual backscatter electron image, quality
  map, phase map, or orientation map, can be used to navigate in. Multiple scans of the
  same size, e.g. a scan of experimental patterns and the best matching simulated
  patterns to that scan, can be plotted simultaneously with HyperSpy's
  ``plot_signals()``.
- Virtual backscatter electron (VBSE) imaging is easily performed with
  ``s.virtual_backscatter_electron_imaging()`` based upon similar functionality in
  pyXem. Arbitrary regions of interests can be used, and the corresponding VBSE image
  can be inspected interactively. Finally, the VBSE image can be obtained in a new
  ``EBSD`` object with ``vbse = s.get_virtual_image()``, before writing the data to an
  image file in your desired format with matplotlib's
  ``imsave('filename.png', vbse.data)``.
- Change scan and pattern size, e.g. by cropping on the detector or extracting a region
  of interest, by using ``s.isig`` or ``s.inav``, respectively. Patterns can be binned
  (upscaled or downscaled) using ``s.rebin``. These methods are provided by HyperSpy.
- Perform static and dynamic background correction by subtraction or division with
  ``s.static_background_correction()`` and ``s.dynamic_background_correction()``. For
  the former correction, relative intensities between patterns can be kept if desired.
- Perform adaptive histogram equalization by setting an appropriate contextual region
  (kernel size) with ``s.adaptive_histogram_equalization()``.
- Rescale pattern intensities to desired data type and range using
  ``s.rescale_intensities()``.
- Multivariate statistical analysis, like principal component analysis and many other
  decomposition algorithms, can be easily performed with ``s.decomposition()``, provided
  by HyperSpy.
- Since the ``EBSD`` class is based upon HyperSpy's ``Signal2D`` class, which itself is
  based upon their ``BaseSignal`` class, all functionality available to ``Signal2D`` is
  also available to the ``EBSD`` class. See HyperSpy's user guide
  (http://hyperspy.org/hyperspy-doc/current/index.html) for details.

Contributors
------------
- Håkon Wiik Ånes
- Tina Bergh

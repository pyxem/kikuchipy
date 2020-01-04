=========
Changelog
=========

0.1.1 (2020-01-04)
==================

This is a bug fix release that ensures that necessary files are uploaded to
PyPI.

0.1.0 (2020-01-04)
==================

We're happy to announce the release of KikuchiPy v0.1.0!

KikuchiPy is an open-source Python library for processing and analysis of
electron backscatter diffraction (EBSD) patterns. The library builds upon the
tools for multi-dimensional data analysis provided by the HyperSpy library.

For more information, a user guide, and the full reference API documentation,
please visit: https://kikuchipy.readthedocs.io

This is the initial pre-release, where things start to get serious... seriously
fun!

Features
--------

* Load EBSD patterns and metadata from the NORDIF binary format (.dat), or
  Bruker Nano's or EDAX TSL's h5ebsd formats (.h5) into an ``EBSD`` object, e.g.
  ``s``, based upon HyperSpy's `Signal2D` class, using ``s = kp.load()``. This
  ensures easy access to patterns and metadata in the attributes ``s.data`` and
  ``s.metadata``, respectively.

* Save EBSD patterns to the NORDIF binary format (.dat) and our own h5ebsd
  format (.h5), using ``s.save()``. Both formats are readable by EMsoft's NORDIF
  and EMEBSD readers, respectively.

* All functionality in KikuchiPy can be performed both directly and lazily
  (except some multivariate analysis algorithms). The latter means that all
  operations on a scan, including plotting, can be done by loading only
  necessary parts of the scan into memory at a time. Ultimately, this lets us
  operate on scans larger than memory using all of our cores.

* Visualize patterns easily with HyperSpy's powerful and versatile ``s.plot()``.
  Any image of the same navigation size, e.g. a virtual backscatter electron
  image, quality map, phase map, or orientation map, can be used to navigate in.
  Multiple scans of the same size, e.g. a scan of experimental patterns and the
  best matching simulated patterns to that scan, can be plotted simultaneously
  with HyperSpy's ``plot_signals()``.

* Virtual backscatter electron (VBSE) imaging is easily performed with
  ``s.virtual_backscatter_electron_imaging()`` based upon similar functionality
  in pyXem. Arbitrary regions of interests can be used, and the corresponding
  VBSE image can be inspected interactively. Finally, the VBSE image can be
  obtained in a new ``EBSD`` object with ``vbse = s.get_virtual_image()``,
  before writing the data to an image file in your desired format with
  matplotlib's ``imsave('filename.png', vbse.data)``.

* Change scan and pattern size, e.g. by cropping on the detector or extracting
  a region of interest, by using ``s.isig`` or ``s.inav``, respectively.
  Patterns can be binned (upscaled or downscaled) using ``s.rebin``. These
  methods are provided by HyperSpy.

* Perform static and dynamic background correction by subtraction or division
  with ``s.static_background_correction()`` and
  ``s.dynamic_background_correction()``. For the former connection, relative
  intensities between patterns can be kept if desired.

* Perform adaptive histogram equalization by setting an appropriate contextual
  region (kernel size) with ``s.adaptive_histogram_equalization()``.

* Rescale pattern intensities to desired data type and range using
  ``s.rescale_intensities()``.

* Multivariate statistical analysis, like principal component analysis and many
  other decomposition algorithms, can be easily performed with
  ``s.decomposition()``, provided by HyperSpy.

* Since the ``EBSD`` class is based upon HyperSpy's ``Signal2D`` class, which
  itself is based upon their ``BaseSignal`` class, all functionality available
  to ``Signal2D`` is also available to the ``EBSD`` class. See HyperSpy's user
  guide (http://hyperspy.org/hyperspy-doc/current/user_guide/tools.html) for
  details.

Contributors to this release (alphabetical by first name)
---------------------------------------------------------

* Håkon Wiik Ånes
* Tina Bergh

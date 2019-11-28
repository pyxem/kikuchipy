==================
Metadata structure
==================

The :py:class:`~kikuchipy.signals.ebsd.EBSD` class stores metadata in the
``metadata`` attribute provided by HyperSpy. While KikuchiPy's metadata
structure (:py:func:`~kikuchipy.util.io.kikuchipy_metadata`) is based on
`HyperSpy's metadata structure
<http://hyperspy.org/hyperspy-doc/current/user_guide/metadata_structure.html>`_,
it includes the two extra nodes ``Acquisition_instrument.SEM.Detector.EBSD`` and
``Acquisition_instrument.Sample.Phases``. The following metadata structure is,
along with the patterns, saved to file when saving an
:py:class:`~kikuchipy.signals.ebsd.EBSD` object in the :ref:`KikuchiPy h5ebsd
format <h5ebsd-format>`.

::

    ├── Acquisition_instrument
    │   └── SEM
    │       ├── Detector
    │       │   └── EBSD
    │       │       ├── azimuth_angle [º]
    │       │       ├── binning
    │       │       ├── detector
    │       │       ├── elevation_angle [º]
    │       │       ├── exposure_time [s]
    │       │       ├── frame_number
    │       │       ├── frame_rate [1/s]
    │       │       ├── gain [dB]
    │       │       ├── grid_type
    │       │       ├── manufacturer
    │       │       ├── sample_tilt [º]
    │       │       ├── scan_time [s]
    │       │       ├── static_background (numpy.ndarray)
    │       │       ├── version
    │       │       ├── xpc
    │       │       ├── ypc
    │       │       └── zpc
    │       ├── beam_energy [kV]
    │       ├── magnification
    │       ├── microscope
    │       └── working_distance [mm]
    └── Sample
        └── Phases
            └── 1
                ├── atom_coordinates
                │   └── 1
                │       ├── atom
                │       ├── coordinates (x0, y0, z0)
                │       ├── debye_waller_factor [nm^2]
                │       └── site_occupation
                ├── formula
                ├── info
                ├── lattice_constants (a, b, c and alfa, beta, gamma) [nm and º]
                ├── laue_group
                ├── material_name
                ├── point_group
                ├── setting
                ├── space_group
                └── symmetry


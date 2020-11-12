==================
Metadata structure
==================

The :class:`~kikuchipy.signals.EBSD` class stores metadata in the
``metadata`` attribute provided by HyperSpy. While kikuchipy's EBSD
(:func:`~kikuchipy.signals.util.ebsd_metadata`) metadata structure is based on
`HyperSpy's metadata structure
<http://hyperspy.org/hyperspy-doc/current/user_guide/metadata_structure.html>`_,
it includes the nodes ``Acquisition_instrument.Sample.Phases`` to store
phase information and ``Acquisition_instrument.SEM.Detector.EBSD`` for
acquisition information. The information in these nodes are written, along with
the patterns, to file when saving an EBSD signal in the
`kikuchipy h5ebsd format <load_save_data.ipynb#h5ebsd>`_.

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
                ├── source
                ├── space_group
                └── symmetry

The utility function :func:`~kikuchipy.signals.util.metadata_nodes` returns the
node strings for the ``SEM`` and ``EBSD`` nodes for convenience.

.. note::

    If you regularly use information relevant to EBSD data not included in the
    metadata structure, you can request this in our `issue tracker
    <https://github.com/pyxem/kikuchipy/issues>`_.

EBSD
====

This node contains information relevant for EBSD data. All parameters can be
set with the method :meth:`~kikuchipy.signals.EBSD.set_experimental_parameters`.
An explanation of each parameter is given in the method's docstring.

Phases
======

This node contains information relevant for EBSD scans or simulated patterns'
phases. All parameters can be set with the :class:`~kikuchipy.signals.EBSD`
method :meth:`~kikuchipy.signals.EBSD.set_phase_parameters`. An explanation of
each parameter is given in the methods' docstring.

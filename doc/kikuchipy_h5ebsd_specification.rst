================================
kikuchipy's h5ebsd specification
================================

This document details the specification for the kikuchipy h5ebsd (HDF5) file
format, which has changed over time. It is based on the h5ebsd file format
presented in :cite:`jackson2014h5ebsd`.

The kikuchipy version's listed below denote when the file format changed, and
shows the HDF5 data group and data set names, their type, array shapes, and a
comment stating units or whether there can be more of them each group or set.
A NumPy array can have any number data type (uint8, float32, etc.) when its type
is not explicitly stated.

0.4.0
=====

The format was updated to include crystallographic information stored in the
orix :class:`~orix.crystal_map.crystal_map.CrystalMap` class in the
:attr:`kikuchipy.signals.EBSD.xmap` attribute.

::

    ├── manufacturer: str, always "kikuchipy"
    ├── version: str, e.g. "0.4.0"
    └── Scan 1: can be more of these, e.g. named "Scan 2" etc.
        ├── EBSD
        │   ├── Data
        │   │   ├── patterns: numpy.ndarray, (n points, n detector rows, n detector columns)
        │   │   └── CrystalMap
        │   │       ├── manufacturer: str, always "orix"
        │   │       ├── version: str, e.g. "0.5.1"
        │   │       └── crystal_map
        │   │           ├── data
        │   │           │   ├── Phi: numpy.ndarray of float, (n points,), in radians
        │   │           │   ├── id: numpy.ndarray of int, (n points,)
        │   │           │   ├── is_in_data: numpy.ndarray of bool, (n points,)
        │   │           │   ├── phase_id: numpy.ndarray of int, (n points,)
        │   │           │   ├── phi1: numpy.ndarray of float, (n points,), in radians
        │   │           │   ├── phi2: numpy.ndarray of float, (n points,), in radians
        │   │           │   ├── x: numpy.ndarray of float, (n points,)
        │   │           │   ├── y: numpy.ndarray of float if 2D or 3D, int if 1D, (n points,) or int
        │   │           │   ├── z: numpy.ndarray of float if 2D or 3D, int if 1D, (n points,) or int
        │   │           │   └── property_array: numpy.ndarray, (n points, ?), can be more of these
        │   │           └── header
        │   │               ├── grid_type: str
        │   │               ├── nx: int
        │   │               ├── ny: int
        │   │               ├── nz: int
        │   │               ├── phases
        │   │               │   └── 0: can be more of these, e.g. named "1" etc.
        │   │               │       ├── color: str
        │   │               │       ├── name: str
        │   │               │       ├── point_group: str
        │   │               │       ├── space_group: int
        │   │               │       └── structure
        │   │               │           ├── atoms
        │   │               │           │   └── 0: can be more of these, e.g. named "1" etc.
        │   │               │           │       ├── U: numpy.ndarray of float, (3, 3)
        │   │               │           │       ├── element: int
        │   │               │           │       ├── label: str
        │   │               │           │       ├── occupancy: float
        │   │               │           │       └── xyz: numpy.ndarray of float, (3,)
        │   │               │           └── lattice
        │   │               │               ├── abcABG: numpy.ndarray of float, (6,), angles in degrees
        │   │               │               └── baserot: numpy.ndarray of float, (3, 3)
        │   │               ├── rotations_per_point: int
        │   │               ├── scan_unit: str
        │   │               ├── x_step: float
        │   │               ├── y_step: float
        │   │               └── z_step: float
        │   └── Header
        │       ├── Detector
        │       │   ├── azimuth_angle: float
        │       │   ├── binning: float
        │       │   ├── exposure_time: float
        │       │   ├── frame_number: int
        │       │   ├── frame_rate: float
        │       │   ├── gain: float
        │       │   ├── name: str
        │       │   ├── pc: numpy.ndarray of float, navigation shape + (3,) or just (3,)
        │       │   ├── px_size: float
        │       │   ├── sample_tilt: float in degrees
        │       │   └── tilt: float in degrees, detector tilt/elevation angle
        │       ├── scan_time: float
        │       └── static_background: numpy.ndarray, (n detector rows, n detector columns)
        └── SEM
            └── Header
                ├── beam_energy: float, in kV
                ├── magnification: int
                ├── microscope: str
                └── working_distance: float

0.1.0
=====

::

    ├── manufacturer: str, always "kikuchipy"
    ├── version: str, e.g. "0.2.0"
    └── Scan 1: can be more of these, e.g. named "Scan 2" etc.
        ├── EBSD
        │   ├── Data
        │   │   ├── patterns: numpy.ndarray, (n points, n detector rows, n detector columns)
        │   │   ├── x_sample: numpy.ndarray of float, (n points,)
        │   │   └── y_sample: numpy.ndarray of float, (n points,)
        │   └── Header
        │       ├── azimuth_angle: float in degrees
        │       ├── binning: int
        │       ├── detector: str
        │       ├── detector_pixel_size: float
        │       ├── elevation_angle: float in degrees
        │       ├── exposure_time: float
        │       ├── frame_number: int
        │       ├── frame_rate: float
        │       ├── gain: float
        │       ├── grid_type: str
        │       ├── n_columns: int
        │       ├── n_rows: int
        │       ├── pattern_height: int
        │       ├── pattern_width: int
        │       ├── sample_tilt: float in degrees
        │       ├── scan_time: float
        │       ├── static_background: numpy.ndarray, (n detector rows, n detector columns)
        │       ├── step_x: float
        │       ├── step_y: float
        │       ├── xpc: float
        │       ├── ypc: float
        │       ├── zpc: float
        │       └── Phases: can be more of these, e.g. named "2" etc.
        │           └── 1
        │               ├── formula: str
        │               ├── info: str
        │               ├── lattice_constants: numpy.ndarray of float, (6,), angles in degrees
        │               ├── laue_group: str
        │               ├── material_name: str
        │               ├── point_group: str
        │               ├── setting: int
        │               ├── source: str
        │               ├── space_group: int
        │               ├── symmetry: str
        │               └── atom_coordinates
        │                   └── 1: can be more of these, e.g. named "2" etc.
        │                       ├── atom: str
        │                       ├── coordinates: numpy.ndarray of float, (3,)
        │                       ├── debye_waller_factor: float
        │                       └── site_occupation: float
        └── SEM
            └── Header
                ├── beam_energy: float, in kV
                ├── magnification: int
                ├── microscope: str
                └── working_distance: float

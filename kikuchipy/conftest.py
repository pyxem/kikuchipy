# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.

import gc
from numbers import Number
import os
from packaging import version
import tempfile

import dask.array as da
from diffpy.structure import Atom, Lattice, Structure
from diffsims.crystallography import ReciprocalLatticePoint
from hyperspy import __version__ as hs_version
from hyperspy.misc.utils import DictionaryTreeBrowser
import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import CrystalMap, create_coordinate_arrays, Phase, PhaseList
from orix.quaternion.rotation import Rotation
from orix.vector import Vector3d, neo_euler
import pytest

import kikuchipy as kp
from kikuchipy.projections.ebsd_projections import (
    detector2reciprocal_lattice,
    detector2direct_lattice,
)


# ------------------------- Helper functions ------------------------- #


def assert_dictionary(dict1, dict2):
    """Assert that two dictionaries are (almost) equal.

    Used to compare signal's axes managers or metadata in tests.
    """
    if isinstance(dict1, DictionaryTreeBrowser):
        dict1 = dict1.as_dictionary()
        dict2 = dict2.as_dictionary()
    for key in dict2.keys():
        if key in ["is_binned", "binned"] and version.parse(hs_version) > version.parse(
            "1.6.2"
        ):  # pragma: no cover
            continue
        if isinstance(dict2[key], dict):
            assert_dictionary(dict1[key], dict2[key])
        else:
            if isinstance(dict2[key], list) or isinstance(dict1[key], list):
                dict2[key] = np.array(dict2[key])
                dict1[key] = np.array(dict1[key])
            if isinstance(dict2[key], (np.ndarray, Number)):
                assert np.allclose(dict1[key], dict2[key])
            else:
                assert dict1[key] == dict2[key]


# ------------------------------ Setup ------------------------------ #


def pytest_sessionstart(session):  # pragma: no cover
    _ = kp.data.nickel_ebsd_large(allow_download=True)


# ----------------------------- Fixtures ----------------------------- #


@pytest.fixture
def dummy_signal():
    """Dummy signal of shape <(3, 3)|(3, 3)>. If this is changed, all
    tests using this signal will fail since they compare the output from
    methods using this signal (as input) to hard-coded outputs.
    """
    # fmt: off
    dummy_array = np.array(
        [
            5, 6, 5, 7, 6, 5, 6, 1, 0, 9, 7, 8, 7, 0, 8, 8, 7, 6, 0, 3, 3, 5, 2,
            9, 3, 3, 9, 8, 1, 7, 6, 4, 8, 8, 2, 2, 4, 0, 9, 0, 1, 0, 2, 2, 5, 8,
            6, 0, 4, 7, 7, 7, 6, 0, 4, 1, 6, 3, 4, 0, 1, 1, 0, 5, 9, 8, 4, 6, 0,
            2, 9, 2, 9, 4, 3, 6, 5, 6, 2, 5, 9
        ],
        dtype=np.uint8
    ).reshape((3, 3, 3, 3))
    # fmt: on
    s = kp.signals.EBSD(dummy_array)
    s.axes_manager.navigation_axes[1].name = "x"
    s.axes_manager.navigation_axes[0].name = "y"
    yield s


@pytest.fixture
def dummy_background():
    """Dummy static background image for the dummy signal. If this is
    changed, all tests using this background will fail since they
    compare the output from methods using this background (as input) to
    hard-coded outputs.
    """
    yield np.array([5, 4, 5, 4, 3, 4, 4, 4, 3], dtype=np.uint8).reshape((3, 3))


@pytest.fixture(params=[[(3, 3), (3, 3), False, np.float32]])
def ebsd_with_axes_and_random_data(request):
    """EBSD signal with minimally defined axes and random data.

    Parameters
    ----------
    navigation_shape : tuple
    signal_shape : tuple
    lazy : bool
    dtype : numpy.dtype
    """
    nav_shape, sig_shape, lazy, dtype = request.param
    nav_ndim = len(nav_shape)
    sig_ndim = len(sig_shape)
    data_shape = nav_shape + sig_shape
    axes = []
    if nav_ndim == 1:
        axes.append(dict(name="x", size=nav_shape[0], scale=1))
    if nav_ndim == 2:
        axes.append(dict(name="y", size=nav_shape[0], scale=1))
        axes.append(dict(name="x", size=nav_shape[1], scale=1))
    if sig_ndim == 2:
        axes.append(dict(name="dy", size=sig_shape[0], scale=1))
        axes.append(dict(name="dx", size=sig_shape[1], scale=1))
    if lazy:
        data = da.random.random(data_shape).astype(dtype)
        yield kp.signals.LazyEBSD(data, axes=axes)
    else:
        data = np.random.random(data_shape).astype(dtype)
        yield kp.signals.EBSD(data, axes=axes)


@pytest.fixture(params=["h5"])
def save_path_hdf5(request):
    """Temporary file in a temporary directory for use when tests need
    to write, and sometimes read again, a signal to, and from, a file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "patterns_temp." + request.param)
        yield file_path
        gc.collect()


@pytest.fixture
def nickel_structure():
    """A diffpy.structure with a Nickel crystal structure."""
    yield Structure(
        atoms=[Atom("Ni", [0, 0, 0])],
        lattice=Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90),
    )


@pytest.fixture
def nickel_phase(nickel_structure):
    """A orix.crystal_map.Phase with a Nickel crystal structure and
    symmetry operations.
    """
    yield Phase(name="ni", structure=nickel_structure, space_group=225)


@pytest.fixture(params=[[[1, 1, 1], [2, 0, 0], [2, 2, 0]]])
def nickel_rlp(request, nickel_phase):
    """A set of reciprocal lattice points for a Nickel crystal
    structure with a minimum interplanar spacing.
    """
    yield ReciprocalLatticePoint(phase=nickel_phase, hkl=request.param)


@pytest.fixture
def pc1():
    """One projection center (PC) in TSL convention."""
    yield [0.4210, 0.7794, 0.5049]


@pytest.fixture(params=[[(1,), (60, 60)]])
def detector(request, pc1):
    """An EBSD detector of a given shape with a number of PCs given by
    a navigation shape.
    """
    nav_shape, sig_shape = request.param
    yield kp.detectors.EBSDDetector(
        shape=sig_shape,
        binning=8,
        px_size=70,
        pc=np.ones(nav_shape + (3,)) * pc1,
        sample_tilt=70,
        tilt=0,
        convention="tsl",
    )


@pytest.fixture
def nickel_rotations():
    """A set of 25 rotations in a TSL crystal reference frame (RD-TD-ND).

    The rotations are from an EMsoft indexing of patterns in the region
    of interest (row0:row1, col0:col1) = (79:84, 134:139) of the first
    Nickel data set in this set of scans:
    https://zenodo.org/record/3265037.
    """
    yield Rotation(
        np.array(
            [
                [0.8662, 0.2033, -0.3483, -0.2951],
                [0.8888, 0.3188, -0.2961, -0.1439],
                [0.8883, 0.3188, -0.2973, -0.1444],
                [0.8884, 0.3187, -0.2975, -0.1437],
                [0.9525, 0.1163, -0.218, -0.1782],
                [0.8658, 0.2031, -0.3486, -0.296],
                [0.8661, 0.203, -0.3486, -0.2954],
                [0.8888, 0.3179, -0.297, -0.1439],
                [0.9728, -0.1634, 0.0677, 0.1494],
                [0.9526, 0.1143, -0.2165, -0.1804],
                [0.8659, 0.2033, -0.3483, -0.2958],
                [0.8663, 0.2029, -0.348, -0.2955],
                [0.8675, 0.1979, -0.3455, -0.298],
                [0.9728, -0.1633, 0.0685, 0.1494],
                [0.9726, -0.1634, 0.0684, 0.1506],
                [0.8657, 0.2031, -0.3481, -0.297],
                [0.8666, 0.2033, -0.3475, -0.2949],
                [0.9111, 0.3315, -0.1267, -0.2095],
                [0.9727, -0.1635, 0.0681, 0.1497],
                [0.9727, -0.1641, 0.0682, 0.1495],
                [0.8657, 0.2024, -0.3471, -0.2986],
                [0.9109, 0.3318, -0.1257, -0.2105],
                [0.9113, 0.3305, -0.1257, -0.2112],
                [0.9725, -0.1643, 0.0691, 0.1497],
                [0.9727, -0.1633, 0.0685, 0.1499],
            ]
        )
    )


@pytest.fixture
def r_tsl2bruker():
    """A rotation from the TSL to Bruker crystal reference frame."""
    yield Rotation.from_neo_euler(
        neo_euler.AxAngle.from_axes_angles(Vector3d.zvector(), np.pi / 2)
    )


@pytest.fixture
def nickel_ebsd_simulation_generator(nickel_phase, detector, nickel_rotations):
    """Generator for EBSD simulations of Kikuchi bands for the Nickel
    data set referenced above.
    """
    yield kp.generators.EBSDSimulationGenerator(
        detector=detector, phase=nickel_phase, rotations=nickel_rotations
    )


@pytest.fixture
def nickel_kikuchi_band(nickel_rlp, nickel_rotations, pc1):
    rlp = nickel_rlp.symmetrise()

    phase = rlp.phase
    hkl = rlp.hkl.data

    nav_shape = (5, 5)

    detector = kp.detectors.EBSDDetector(
        shape=(60, 60),
        binning=8,
        px_size=70,
        pc=np.ones(nav_shape + (3,)) * pc1,
        sample_tilt=70,
        tilt=0,
        convention="tsl",
    )

    nav_dim = detector.navigation_dimension
    navigation_axes = (1, 2)[:nav_dim]

    # Output shape is (3, n, 3) or (3, ny, nx, 3)
    det2recip = detector2reciprocal_lattice(
        sample_tilt=detector.sample_tilt,
        detector_tilt=detector.tilt,
        lattice=phase.structure.lattice,
        rotation=nickel_rotations.reshape(*nav_shape),
    )

    # Output shape is (nhkl, n, 3) or (nhkl, ny, nx, 3)
    hkl_detector = np.tensordot(hkl, det2recip, axes=(1, 0))
    # Determine whether a band is visible in a pattern
    upper_hemisphere = hkl_detector[..., 2] > 0
    is_in_some_pattern = np.sum(upper_hemisphere, axis=navigation_axes) != 0
    # Get bands that are in some pattern and their coordinates in the
    # proper shape
    hkl = hkl[is_in_some_pattern, ...]
    hkl_in_pattern = upper_hemisphere[is_in_some_pattern, ...].T
    hkl_detector = np.moveaxis(
        hkl_detector[is_in_some_pattern], source=0, destination=nav_dim
    )

    yield kp.simulations.features.KikuchiBand(
        phase=phase,
        hkl=hkl,
        hkl_detector=hkl_detector,
        in_pattern=hkl_in_pattern,
        gnomonic_radius=detector.r_max,
    )


@pytest.fixture
def nickel_zone_axes(nickel_kikuchi_band, nickel_rotations, pc1):
    bands = nickel_kikuchi_band
    hkl = bands.hkl.data
    phase = bands.phase

    nav_shape = (5, 5)

    detector = kp.detectors.EBSDDetector(
        shape=(60, 60),
        binning=8,
        px_size=70,
        pc=np.ones(nav_shape + (3,)) * pc1,
        sample_tilt=70,
        tilt=0,
        convention="tsl",
    )

    nav_dim = detector.navigation_dimension
    navigation_axes = (1, 2)[:nav_dim]

    n_hkl = bands.size
    n_hkl2 = n_hkl ** 2
    uvw = np.cross(hkl[:, np.newaxis, :], hkl).reshape((n_hkl2, 3))
    not000 = np.count_nonzero(uvw, axis=1) != 0
    uvw = uvw[not000]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = uvw / np.gcd.reduce(uvw, axis=1)[:, np.newaxis]
    uvw = np.unique(uvw, axis=0).astype(int)
    det2direct = detector2direct_lattice(
        sample_tilt=detector.sample_tilt,
        detector_tilt=detector.tilt,
        lattice=phase.structure.lattice,
        rotation=nickel_rotations.reshape(*nav_shape),
    )
    uvw_detector = np.tensordot(uvw, det2direct, axes=(1, 0))
    upper_hemisphere = uvw_detector[..., 2] > 0
    is_in_some_pattern = np.sum(upper_hemisphere, axis=navigation_axes) != 0
    uvw = uvw[is_in_some_pattern, ...]
    uvw_in_pattern = upper_hemisphere[is_in_some_pattern, ...].T
    uvw_detector = np.moveaxis(
        uvw_detector[is_in_some_pattern], source=0, destination=nav_dim
    )

    yield kp.simulations.features.ZoneAxis(
        phase=phase,
        uvw=uvw,
        uvw_detector=uvw_detector,
        in_pattern=uvw_in_pattern,
        gnomonic_radius=detector.r_max,
    )


@pytest.fixture
def rotations():
    return Rotation([(2, 4, 6, 8), (-1, -3, -5, -7)])


@pytest.fixture
def get_single_phase_xmap(rotations):
    def _get_single_phase_xmap(
        nav_shape,
        rotations_per_point=5,
        prop_names=["scores", "simulation_indices"],
        name="a",
        phase_id=0,
        step_sizes=None,
    ):
        d, map_size = create_coordinate_arrays(shape=nav_shape, step_sizes=step_sizes)
        rot_idx = np.random.choice(
            np.arange(rotations.size), map_size * rotations_per_point
        )
        data_shape = (map_size,)
        if rotations_per_point > 1:
            data_shape += (rotations_per_point,)
        d["rotations"] = rotations[rot_idx].reshape(*data_shape)
        d["phase_id"] = np.ones(map_size) * phase_id
        d["phase_list"] = PhaseList(Phase(name=name))
        # Scores and simulation indices
        d["prop"] = {
            prop_names[0]: np.ones(data_shape, dtype=np.float32),
            prop_names[1]: np.arange(np.prod(data_shape)).reshape(data_shape),
        }
        return CrystalMap(**d)

    return _get_single_phase_xmap


@pytest.fixture(params=[((2, 3), (60, 60), np.uint8, 2, False, True)])
def oxford_binary_file(tmpdir, request):
    """Create a dummy Oxford Instruments' binary .ebsp file.

    The creation of a dummy .ebsp file is explained in more detail in
    kikuchipy/data/oxford_binary/create_dummy_oxford_binary_file.py.

    Parameters expected in `request`
    -------------------------------
    navigation_shape : tuple of ints
    signal_shape : tuple of ints
    dtype : numpy.dtype
    version : int
    compressed : bool
    all_present : bool
    """
    # Unpack parameters
    (nr, nc), (sr, sc), dtype, ver, compressed, all_present = request.param

    fname = tmpdir.join("dummy_oxford_file.ebsp")
    f = open(fname, mode="w")

    if ver != 0:
        np.array(-ver, dtype=np.int64).tofile(f)

    pattern_header_size = 16
    if ver == 0:
        pattern_footer_size = 0
    elif ver == 1:
        pattern_footer_size = 16
    else:
        pattern_footer_size = 18

    n_patterns = nr * nc
    n_pixels = sr * sc

    if np.issubdtype(dtype, np.uint8):
        n_bytes = n_pixels
    else:
        n_bytes = 2 * n_pixels

    pattern_starts = np.arange(n_patterns, dtype=np.int64)
    pattern_starts *= pattern_header_size + n_bytes + pattern_footer_size
    pattern_starts += n_patterns * 8
    if ver != 0:
        pattern_starts += 8

    pattern_starts = np.roll(pattern_starts, shift=1)
    if not all_present:
        pattern_starts[0] = 0
    pattern_starts.tofile(f)
    new_order = np.roll(np.arange(n_patterns), shift=-1)

    pattern_header = np.array([compressed, sr, sc, n_bytes], dtype=np.int32)
    data = np.arange(n_patterns * n_pixels, dtype=dtype).reshape((nr, nc, sr, sc))

    if not all_present:
        new_order = new_order[1:]

    for i in new_order:
        r, c = np.unravel_index(i, (nr, nc))
        pattern_header.tofile(f)
        data[r, c].tofile(f)
        if ver > 1:
            np.array(1, dtype=bool).tofile(f)  # has_beam_x
        if ver > 0:
            np.array(c, dtype=np.float64).tofile(f)  # beam_x
        if ver > 1:
            np.array(1, dtype=bool).tofile(f)  # has_beam_y
        if ver > 0:
            np.array(r, dtype=np.float64).tofile(f)  # beam_y

    f.close()

    yield f


# ---------------------- pytest doctest-modules ---------------------- #


@pytest.fixture(autouse=True)
def doctest_setup_teardown(request):
    # Setup
    plt.ioff()  # Interactive plotting off
    temporary_directory = tempfile.TemporaryDirectory()
    original_directory = os.getcwd()
    os.chdir(temporary_directory.name)
    yield

    # Teardown
    os.chdir(original_directory)
    temporary_directory.cleanup()
    plt.close("all")


@pytest.fixture(autouse=True)
def import_to_namespace(doctest_namespace):
    dir_path = os.path.dirname(__file__)
    doctest_namespace["DATA_DIR"] = os.path.join(dir_path, "data/kikuchipy_h5ebsd")

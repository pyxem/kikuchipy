# Copyright 2019-2022 The kikuchipy developers
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
import warnings

import dask.array as da
from diffpy.structure import Atom, Lattice, Structure
import hyperspy.api as hs
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import CrystalMap, create_coordinate_arrays, Phase, PhaseList
from orix.quaternion import Rotation
from orix.vector import Vector3d
import pytest

import kikuchipy as kp


if kp._pyvista_installed:
    import pyvista as pv

    pv.OFF_SCREEN = True
    pv.global_theme.interactive = False


warnings.filterwarnings("always", category=DeprecationWarning)


# ------------------------- Helper functions ------------------------- #


def assert_dictionary(dict1, dict2):
    """Assert that two dictionaries are (almost) equal.

    Used to compare signal's axes managers or metadata in tests.
    """
    for key in dict2.keys():
        if key in ["is_binned", "binned"] and version.parse(
            hs.__version__
        ) > version.parse(
            "1.6.2"
        ):  # pragma: no cover
            continue
        if isinstance(dict2[key], dict):
            assert_dictionary(dict1[key], dict2[key])
        else:
            if isinstance(dict2[key], list) or isinstance(
                dict1[key], list
            ):  # pragma: no cover
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
def dummy_signal(dummy_background):
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
    s = kp.signals.EBSD(dummy_array, static_background=dummy_background)
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
    data_size = int(np.prod(data_shape))
    axes = []
    if nav_ndim == 1:
        axes.append(dict(name="x", size=nav_shape[0], scale=1))
    if nav_ndim == 2:
        axes.append(dict(name="y", size=nav_shape[0], scale=1))
        axes.append(dict(name="x", size=nav_shape[1], scale=1))
    if sig_ndim == 2:
        axes.append(dict(name="dy", size=sig_shape[0], scale=1))
        axes.append(dict(name="dx", size=sig_shape[1], scale=1))
    if np.issubdtype(dtype, np.integer):
        data_kwds = dict(low=1, high=255, size=data_size)
    else:
        data_kwds = dict(low=0.1, high=1, size=data_size)
    if lazy:
        data = da.random.uniform(**data_kwds).reshape(data_shape).astype(dtype)
        yield kp.signals.LazyEBSD(data, axes=axes)
    else:
        data = np.random.uniform(**data_kwds).reshape(data_shape).astype(dtype)
        yield kp.signals.EBSD(data, axes=axes)


@pytest.fixture(params=["h5"])
def save_path_hdf5(request):
    """Temporary file in a temporary directory for use when tests need
    to write, and sometimes read again, a signal to, and from, a file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.join(tmp, "patterns." + request.param)
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
def r_tsl2bruker():
    """A rotation from the TSL to Bruker crystal reference frame."""
    yield Rotation.from_axes_angles(Vector3d.zvector(), np.pi / 2)


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


@pytest.fixture(params=[(1, (2, 3), (60, 60), "uint8", 2, False)])
def edax_binary_file(tmpdir, request):
    """Create a dummy EDAX binary UP1/2 file.

    The creation of dummy UP1/2 files is explained in more detail in
    kikuchipy/data/edax_binary/create_dummy_edax_binary_file.py.

    Parameters expected in `request`
    -------------------------------
    up_version : int
    navigation_shape : tuple of ints
    signal_shape : tuple of ints
    dtype : str
    version : int
    is_hex : bool
    """
    # Unpack parameters
    up_ver, (ny, nx), (sy, sx), dtype, ver, is_hex = request.param

    if up_ver == 1:
        fname = tmpdir.join("dummy_edax_file.up1")
        file = open(fname, mode="w")

        # File header: 16 bytes
        # 4 bytes with the file version
        np.array([ver], "uint32").tofile(file)
        # 12 bytes with the pattern width, height and file offset position
        np.array([sx, sy, 16], "uint32").tofile(file)

        # Patterns
        np.ones(ny * nx * sy * sx, dtype).tofile(file)
    else:  # up_ver == 2
        fname = tmpdir.join("dummy_edax_file.up2")
        file = open(fname, mode="w")

        # File header: 42 bytes
        # 4 bytes with the file version
        np.array([ver], "uint32").tofile(file)
        # 12 bytes with the pattern width, height and file offset position
        np.array([sx, sy, 42], "uint32").tofile(file)
        # 1 byte with any "extra patterns" (?)
        np.array([1], "uint8").tofile(file)
        # 8 bytes with the map width and height (same as square)
        np.array([nx, ny], "uint32").tofile(file)
        # 1 byte to say whether the grid is hexagonal
        np.array([int(is_hex)], "uint8").tofile(file)
        # 16 bytes with the horizontal and vertical step sizes
        np.array([np.pi, np.pi / 2], "float64").tofile(file)

        # Patterns
        np.ones((ny * nx + ny // 2) * sy * sx, dtype).tofile(file)

    file.close()

    yield file


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


@pytest.fixture
def nickel_ebsd_small_di_xmap():
    """Yield an :class:`~orix.crystal_map.CrystalMap` from dictionary
    indexing of the :func:`kikuchipy.data.nickel_ebsd_small` data set.

    Dictionary indexing was performed with the following script:

    .. code-block:: python

        import kikuchipy as kp
        from orix.sampling import get_sample_fundamental


        s = kp.data.nickel_ebsd_small()
        s.remove_static_background()
        s.remove_dynamic_background()

        mp = kp.data.nickel_ebsd_master_pattern_small(energy=20, projection="lambert")
        rot = get_sample_fundamental(resolution=1.4, point_group=mp.phase.point_group)
        detector = kp.detectors.EBSDDetector(
            shape=s.axes_manager.signal_shape[::-1],
            sample_tilt=70,
            pc=(0.421, 0.7794, 0.5049),
            convention="tsl"
        )
        sim_dict = mp.get_patterns(rotations=rot, detector=detector, energy=20)
        xmap = s.dictionary_indexing(dictionary=sim_dict, keep_n=1)
    """
    coords, _ = create_coordinate_arrays(shape=(3, 3), step_sizes=(1.5, 1.5))
    # fmt: off
    grain1 = (0.9542, -0.0183, -0.2806,  0.1018)
    grain2 = (0.9542,  0.0608, -0.2295, -0.1818)
    xmap = CrystalMap(
        rotations=Rotation((
            grain1, grain2, grain2,
            grain1, grain2, grain2,
            grain1, grain2, grain2
        )),
        x=coords["x"],
        y=coords["y"],
        prop=dict(scores=np.array((
            0.4364652,  0.3772456,  0.4140171,
            0.4537009,  0.37445727, 0.43675864,
            0.42391658, 0.38740265, 0.41931134
        ))),
        phase_list=PhaseList(Phase("ni", 225, "m-3m")),
    )
    # fmt: on
    yield xmap


@pytest.fixture
def ni_kikuchipy_h5ebsd_file(tmp_path, nickel_ebsd_small_di_xmap, detector):
    """Temporary file in kikuchipy's h5ebsd format with a crystal map
    and detector stored.
    """
    s = kp.data.nickel_ebsd_small()
    s.xmap = nickel_ebsd_small_di_xmap
    detector.pc = np.ones((3, 3, 3)) * detector.pc
    s.detector = detector
    fname = tmp_path / "kp_file.h5"
    s.save(fname)
    yield fname


@pytest.fixture
def ni_small_axes_manager():
    """Axes manager for :func:`kikuchipy.data.nickel_ebsd_small`."""
    names = ["y", "x", "dy", "dx"]
    scales = [1.5, 1.5, 1, 1]
    sizes = [3, 3, 60, 60]
    navigates = [True, True, False, False]
    axes_manager = {}
    for i in range(len(names)):
        axes_manager[f"axis-{i}"] = {
            "name": names[i],
            "scale": scales[i],
            "offset": 0.0,
            "size": sizes[i],
            "units": "um",
            "navigate": navigates[i],
        }
    yield axes_manager


@pytest.fixture(params=[("_x{}y{}.tif", (3, 3))])
def ebsd_directory(tmpdir, request):
    """Temporary directory with EBSD files as .tif, .png or .bmp files.

    Parameters expected in `request`
    -------------------------------
    xy_pattern : str
    nav_shape : tuple of ints
    """
    s = kp.data.nickel_ebsd_small()
    s.unfold_navigation_space()

    xy_pattern, nav_shape = request.param
    y, x = np.indices(nav_shape)
    x = x.ravel()
    y = y.ravel()
    for i in range(s.axes_manager.navigation_size):
        fname = os.path.join(tmpdir, "pattern" + xy_pattern.format(x[i], y[i]))
        iio.imwrite(fname, s.data[i])

    yield tmpdir


# ---------------------- pytest doctest-modules ---------------------- #


@pytest.fixture(autouse=True)
def doctest_setup_teardown(request):
    # Setup
    # Temporarily turn off interactive plotting with Matplotlib
    plt.ioff()

    # Temporarily suppress HyperSpy's progressbar
    hs.preferences.General.show_progressbar = False

    # Temporary directory for saving files in
    temporary_directory = tempfile.TemporaryDirectory()
    original_directory = os.getcwd()
    os.chdir(temporary_directory.name)
    yield

    # Teardown
    os.chdir(original_directory)
    temporary_directory.cleanup()


@pytest.fixture(autouse=True)
def import_to_namespace(doctest_namespace):
    dir_path = os.path.dirname(__file__)
    doctest_namespace["DATA_DIR"] = os.path.join(dir_path, "data/kikuchipy_h5ebsd")

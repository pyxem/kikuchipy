# Copyright 2019-2024 The kikuchipy developers
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

# Why is this file located in the top directory and not in tests/?
# Because if it was, running "pytest --doctest-modules src" wouldn't
# discover this file.

from io import TextIOWrapper
from numbers import Number
import os
from pathlib import Path
import tempfile
from typing import Callable, Generator

import dask.array as da
from diffpy.structure import Atom, Lattice, Structure
import h5py
import hyperspy.api as hs
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy.constants import dependency_version
from kikuchipy.data._data import marshall
from kikuchipy.data._dummy_files.bruker_h5ebsd import (
    create_dummy_bruker_h5ebsd_file,
    create_dummy_bruker_h5ebsd_nonrectangular_roi_file,
    create_dummy_bruker_h5ebsd_roi_file,
)
from kikuchipy.data._dummy_files.oxford_h5ebsd import create_dummy_oxford_h5ebsd_file
from kikuchipy.io.plugins._h5ebsd import _dict2hdf5group

if dependency_version["pyvista"] is not None:
    import pyvista as pv

    pv.OFF_SCREEN = True
    pv.global_theme.interactive = False


DATA_PATH = Path(kp.data.__file__).parent.resolve()

# ------------------------------ Setup ------------------------------ #


def pytest_sessionstart(session):
    _ = kp.data.nickel_ebsd_large(allow_download=True)
    plt.rcParams["backend"] = "agg"


# ---------------------- pytest doctest-modules ---------------------- #


@pytest.fixture(autouse=True)
def doctest_setup_teardown(request):
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


@pytest.fixture(autouse=True)
def import_to_namespace(doctest_namespace) -> None:
    doctest_namespace["DATA_PATH"] = DATA_PATH / "kikuchipy_h5ebsd"


# ----------------------------- Fixtures ----------------------------- #


@pytest.fixture
def assert_dictionary_func() -> Callable:
    def assert_dictionary(dict1: dict, dict2: dict) -> None:
        """Assert that two dictionaries are (almost) equal.

        Used to compare signal's axes managers or metadata in tests.
        """
        for key in dict2.keys():
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

    return assert_dictionary


@pytest.fixture
def dummy_signal(
    dummy_background: np.ndarray,
) -> Generator[kp.signals.EBSD, None, None]:
    """Dummy signal of shape <3, 3|3, 3>. If this is changed, all
    tests using this signal will fail since they compare the output from
    methods using this signal (as input) to hard-coded outputs.
    """
    nav_shape = (3, 3)
    nav_size = int(np.prod(nav_shape))
    sig_shape = (3, 3)

    # fmt: off
    dummy_array = np.array(
        [
            5, 6, 5, 7, 6, 5, 6, 1, 0, 9, 7, 8, 7, 0, 8, 8, 7, 6, 0, 3, 3, 5, 2,
            9, 3, 3, 9, 8, 1, 7, 6, 4, 8, 8, 2, 2, 4, 0, 9, 0, 1, 0, 2, 2, 5, 8,
            6, 0, 4, 7, 7, 7, 6, 0, 4, 1, 6, 3, 4, 0, 1, 1, 0, 5, 9, 8, 4, 6, 0,
            2, 9, 2, 9, 4, 3, 6, 5, 6, 2, 5, 9
        ],
        dtype=np.uint8
    ).reshape(nav_shape + sig_shape)
    # fmt: on

    # Initialize and set static background attribute
    s = kp.signals.EBSD(dummy_array, static_background=dummy_background)

    # Axes manager
    s.axes_manager.navigation_axes[1].name = "x"
    s.axes_manager.navigation_axes[0].name = "y"

    # Crystal map
    phase_list = PhaseList([Phase("a", space_group=225), Phase("b", space_group=227)])
    y, x = np.indices(nav_shape)
    s.xmap = CrystalMap(
        rotations=Rotation.identity((nav_size,)),
        # fmt: off
        phase_id=np.array([
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 0],
        ]).ravel(),
        # fmt: on
        phase_list=phase_list,
        x=x.ravel(),
        y=y.ravel(),
    )
    pc = np.arange(np.prod(nav_shape) * 3).reshape(nav_shape + (3,))
    pc = pc.astype(float) / pc.max()
    s.detector = kp.detectors.EBSDDetector(shape=sig_shape, pc=pc)

    yield s


@pytest.fixture
def dummy_background() -> Generator[np.ndarray, None, None]:
    """Dummy static background image for the dummy signal. If this is
    changed, all tests using this background will fail since they
    compare the output from methods using this background (as input) to
    hard-coded outputs.
    """
    yield np.array([5, 4, 5, 4, 3, 4, 4, 4, 3], dtype=np.uint8).reshape((3, 3))


@pytest.fixture(params=[[(3, 3), (3, 3), False, np.float32]])
def ebsd_with_axes_and_random_data(request) -> Generator[kp.signals.EBSD, None, None]:
    """EBSD signal with minimally defined axes and random data.

    Parameters expected in `request`
    -------------------------------
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
        axes.append({"name": "x", "size": nav_shape[0], "scale": 1})
    if nav_ndim == 2:
        axes.append({"name": "y", "size": nav_shape[0], "scale": 1})
        axes.append({"name": "x", "size": nav_shape[1], "scale": 1})
    if sig_ndim == 2:
        axes.append({"name": "dy", "size": sig_shape[0], "scale": 1})
        axes.append({"name": "dx", "size": sig_shape[1], "scale": 1})
    if np.issubdtype(dtype, np.integer):
        kw = {"low": 1, "high": 255, "size": data_size}
    else:
        kw = {"low": 0.1, "high": 1, "size": data_size}
    if lazy:
        data = da.random.uniform(**kw).reshape(data_shape).astype(dtype)
        s = kp.signals.LazyEBSD(data, axes=axes)
    else:
        data = np.random.uniform(**kw).reshape(data_shape).astype(dtype)
        s = kp.signals.EBSD(data, axes=axes)
    yield s


@pytest.fixture
def nickel_structure() -> Generator[Structure, None, None]:
    """A diffpy.structure with a Nickel crystal structure."""
    yield Structure(
        atoms=[Atom("Ni", [0, 0, 0])],
        lattice=Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90),
    )


@pytest.fixture
def nickel_phase(nickel_structure) -> Generator[Phase, None, None]:
    yield Phase(name="ni", structure=nickel_structure, space_group=225)


@pytest.fixture
def pc1() -> Generator[list[float], None, None]:
    """One projection center (PC) in TSL convention."""
    yield [0.4210, 0.7794, 0.5049]


@pytest.fixture(params=[[(1,), (60, 60)]])
def detector(request, pc1) -> Generator[kp.detectors.EBSDDetector, None, None]:
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
def rotations() -> Generator[Rotation, None, None]:
    yield Rotation([(2, 4, 6, 8), (-1, -3, -5, -7)])


@pytest.fixture
def get_single_phase_xmap(rotations) -> Generator[Callable, None, None]:
    def _get_single_phase_xmap(
        nav_shape,
        rotations_per_point=5,
        prop_names=("scores", "simulation_indices"),
        name="a",
        space_group=225,
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
        d["phase_list"] = PhaseList(Phase(name=name, space_group=space_group))
        # Scores and simulation indices
        d["prop"] = {
            prop_names[0]: np.ones(data_shape, dtype=np.float32),
            prop_names[1]: np.arange(np.prod(data_shape)).reshape(data_shape),
        }
        return CrystalMap(**d)

    yield _get_single_phase_xmap


# ---------------------------- IO fixtures --------------------------- #


@pytest.fixture(params=["h5"])
def save_path_hdf5(request, tmpdir) -> Generator[Path, None, None]:
    """Temporary file in a temporary directory for use when tests need
    to write, and sometimes read again, a signal to, and from, a file.
    """
    ext = request.param
    yield Path(tmpdir / f"patterns.{ext}")


@pytest.fixture()
def save_path_nordif(tmpdir) -> Generator[Path, None, None]:
    yield Path(tmpdir / "nordif/save_temp.dat")


@pytest.fixture
def ni_small_axes_manager() -> Generator[dict, None, None]:
    """Axes manager for :func:`kikuchipy.data.nickel_ebsd_small`."""
    names = ["y", "x", "dy", "dx"]
    scales = [1.5, 1.5, 1, 1]
    sizes = [3, 3, 60, 60]
    navigates = [True, True, False, False]
    axes_manager = {}
    for i in range(len(names)):
        axes_manager[f"axis-{i}"] = {
            "_type": "UniformDataAxis",
            "name": names[i],
            "units": "um",
            "navigate": navigates[i],
            "is_binned": False,
            "size": sizes[i],
            "scale": scales[i],
            "offset": 0.0,
        }
    yield axes_manager


@pytest.fixture(params=[("_x{}y{}.tif", (3, 3))])
def ebsd_directory(tmpdir, request) -> Generator[Path, None, None]:
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
        fname = str(tmpdir / ("pattern" + xy_pattern.format(x[i], y[i])))
        iio.imwrite(fname, s.data[i])

    yield tmpdir


# ------------------------ kikuchipy formats ------------------------- #


@pytest.fixture
def kikuchipy_h5ebsd_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "kikuchipy_h5ebsd"


@pytest.fixture
def nickel_ebsd_large_h5ebsd_renamed() -> Generator[Path, None, None]:
    f1 = Path(marshall.path) / "data/nickel_ebsd_large/patterns.h5"
    f2 = f1.rename(f1.with_suffix(".bak"))
    yield f2
    f2.rename(f1)


# --------------------------- EDAX formats --------------------------- #


@pytest.fixture
def edax_binary_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "edax_binary"


@pytest.fixture(params=[(1, (2, 3), (60, 60), "uint8", 2, False)])
def edax_binary_file(tmpdir, request) -> Generator[TextIOWrapper, None, None]:
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
        f = open(fname, mode="w")

        # File header: 16 bytes
        # 4 bytes with the file version
        np.array([ver], "uint32").tofile(f)
        # 12 bytes with the pattern width, height and file offset position
        np.array([sx, sy, 16], "uint32").tofile(f)

        # Patterns
        np.ones(ny * nx * sy * sx, dtype).tofile(f)
    else:  # up_ver == 2
        fname = tmpdir.join("dummy_edax_file.up2")
        f = open(fname, mode="w")

        # File header: 42 bytes
        # 4 bytes with the file version
        np.array([ver], "uint32").tofile(f)
        # 12 bytes with the pattern width, height and file offset position
        np.array([sx, sy, 42], "uint32").tofile(f)
        # 1 byte with any "extra patterns" (?)
        np.array([1], "uint8").tofile(f)
        # 8 bytes with the map width and height (same as square)
        np.array([nx, ny], "uint32").tofile(f)
        # 1 byte to say whether the grid is hexagonal
        np.array([int(is_hex)], "uint8").tofile(f)
        # 16 bytes with the horizontal and vertical step sizes
        np.array([np.pi, np.pi / 2], "float64").tofile(f)

        # Patterns
        np.ones((ny * nx + ny // 2) * sy * sx, dtype).tofile(f)

    f.close()

    yield f


@pytest.fixture
def edax_h5ebsd_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "edax_h5ebsd"


# -------------------- Oxford Instruments formats -------------------- #


@pytest.fixture
def oxford_binary_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "oxford_binary"


@pytest.fixture(params=[((2, 3), (60, 60), np.uint8, 2, False, True)])
def oxford_binary_file(tmpdir, request) -> Generator[TextIOWrapper, None, None]:
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

    if ver > 0:
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
    if ver in [1, 2, 3]:
        pattern_starts += 8
    elif ver > 3:
        np.array(0, dtype=np.uint8).tofile(f)
        pattern_starts += 9

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
        if ver > 4:
            extra_pattern_header = np.array([c, r], dtype=np.int32)
            extra_pattern_header.tofile(f)
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


@pytest.fixture()
def oxford_h5ebsd_file(tmpdir) -> Generator[TextIOWrapper, None, None]:
    fpath = tmpdir / "patterns.h5oina"
    create_dummy_oxford_h5ebsd_file(fpath)
    yield fpath


# -------------------------- EMsoft formats -------------------------- #


@pytest.fixture
def emsoft_ebsd_master_pattern_file() -> Generator[Path, None, None]:
    yield DATA_PATH / "emsoft_ebsd_master_pattern/master_patterns.h5"


@pytest.fixture
def emsoft_ebsd_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "emsoft_ebsd"


@pytest.fixture
def emsoft_ebsd_file(emsoft_ebsd_path) -> Generator[Path, None, None]:
    yield emsoft_ebsd_path / "EBSD_TEST_Ni.h5"


@pytest.fixture
def emsoft_ebsd_master_pattern_metadata() -> Generator[dict, None, None]:
    fname = "master_patterns.h5"
    yield {
        "General": {"original_filename": fname, "title": fname.split(".")[0]},
        "Signal": {"signal_type": "EBSDMasterPattern"},
    }


@pytest.fixture(params=[["hemisphere", "energy", "height", "width"]])
def emsoft_ebsd_master_pattern_axes_manager(request) -> Generator[dict, None, None]:
    axes = request.param
    am = {
        "hemisphere": {
            "name": "hemisphere",
            "scale": 1,
            "offset": 0,
            "size": 2,
            "units": "",
            "navigate": True,
        },
        "energy": {
            "name": "energy",
            "scale": 1,
            "offset": 10.0,
            "size": 11,
            "units": "keV",
            "navigate": True,
        },
        "height": {
            "name": "height",
            "scale": 1,
            "offset": -7.0,
            "size": 13,
            "units": "px",
            "navigate": False,
        },
        "width": {
            "name": "width",
            "scale": 1,
            "offset": -7.0,
            "size": 13,
            "units": "px",
            "navigate": False,
        },
    }
    d = {}
    for i, a in enumerate(axes):
        d["axis-" + str(i)] = am[a]
    yield d


@pytest.fixture
def emsoft_ecp_master_pattern_file(tmpdir) -> Generator[Path, None, None]:
    """Dummy EMsoft ECP master pattern file."""

    fpath = tmpdir / "ecp_master_pattern.h5"
    f = h5py.File(fpath, mode="w")

    npx = 6
    signal_shape = (npx * 2 + 1,) * 2
    energies = np.linspace(10, 20, 11, dtype=np.float32)
    data_shape = (len(energies),) + signal_shape

    mp_lam_upper = np.ones((1,) + data_shape, dtype=np.float32) * energies.reshape(
        (1, 11, 1, 1)
    )
    mp_lam_lower = mp_lam_upper
    circle = kp.filters.Window(shape=signal_shape).astype(np.float32)
    mp_sph_upper = mp_lam_upper.squeeze() * circle
    mp_sph_lower = mp_sph_upper

    data = {
        "CrystalData": {
            "AtomData": np.array(
                [[0.1587, 0], [0.6587, 0], [0, 0.25], [1, 1], [0.005, 0.005]],
                dtype=np.float32,
            ),
            "Atomtypes": np.array([13, 29], dtype=np.int32),
            "CrystalSystem": 2,
            "LatticeParameters": np.array([0.5949, 0.5949, 0.5821, 90, 90, 90]),
            "Natomtypes": 2,
            "Source": "Su Y.C., Yan J., Lu P.T., Su J.T.: Thermodynamic...",
            "SpaceGroupNumber": 140,
            "SpaceGroupSetting": 1,
        },
        "EMData": {
            "ECPmaster": {
                "EkeV": np.linspace(10, 20, 11, dtype=np.float32),
                "mLPNH": mp_lam_upper,  # mLPSH written below
                "masterSPNH": mp_sph_upper,
                "masterSPSH": mp_sph_lower,
                "numset": 1,
            }
        },
        "NMLparameters": {
            "ECPMasterNameList": {"dmin": 0.05, "npx": npx},
            "MCCLNameList": {
                "Ebinsize": energies[1] - energies[0],
                "Ehistmin": np.min(energies),
                "EkeV": np.max(energies),
                "MCmode": "CSDA",
                "dataname": "crystal_data/al2cu/al2cu_mc_mp_20kv.h5",
                "depthmax": 100.0,
                "depthstep": 1.0,
                "mode": "bse1",
                "numsx": npx,
                "sigend": 10.0,
                "sigstart": 0.0,
                "sigstep": 2.0,
                "totnum_el": 2000000000,
            },
            "BetheList": {"c1": 4.0, "c2": 8.0, "c3": 50.0, "sgdbdiff": 1.0},
        },
        "EMheader": {
            "ECPmaster": {"ProgramName": np.array([b"EMECPmaster.f90"], dtype="S15")},
        },
    }

    _dict2hdf5group(dictionary=data, group=f["/"])

    # One chunked data set
    f["EMData/ECPmaster"].create_dataset("mLPSH", data=mp_lam_lower, chunks=True)

    # One byte string with latin-1 stuff
    creation_time = b"12:30:13.559 PM\xf0\x14\x1e\xc8\xbcU"
    f["CrystalData"].create_dataset("CreationTime", data=creation_time)

    f.close()

    yield fpath


@pytest.fixture
def emsoft_tkd_master_pattern_file(tmpdir) -> Generator[Path, None, None]:
    fpath = tmpdir / "tkd_master_pattern.h5"
    f = h5py.File(fpath, mode="w")

    npx = 6
    signal_shape = (npx * 2 + 1,) * 2
    energies = np.linspace(10, 20, 11, dtype=np.float32)
    data_shape = (len(energies),) + signal_shape

    mp_lam_upper = np.ones((1,) + data_shape, dtype=np.float32) * energies.reshape(
        (1, 11, 1, 1)
    )
    mp_lam_lower = mp_lam_upper
    circle = kp.filters.Window(shape=signal_shape).astype(np.float32)
    mp_sph_upper = mp_lam_upper.squeeze() * circle
    mp_sph_lower = mp_sph_upper

    data = {
        "CrystalData": {
            "AtomData": np.array(
                [[0.1587, 0], [0.6587, 0], [0, 0.25], [1, 1], [0.005, 0.005]],
                dtype=np.float32,
            ),
            "Atomtypes": np.array([13, 29], dtype=np.int32),
            "CrystalSystem": 2,
            "LatticeParameters": np.array([0.5949, 0.5949, 0.5821, 90, 90, 90]),
            "Natomtypes": 2,
            "Source": "Su Y.C., Yan J., Lu P.T., Su J.T.: Thermodynamic...",
            "SpaceGroupNumber": 140,
            "SpaceGroupSetting": 1,
        },
        "EMData": {
            "TKDmaster": {
                "BetheParameters": np.array([4, 8, 50, 1], dtype=np.float32),
                "EkeVs": np.linspace(10, 20, 11, dtype=np.float32),
                "mLPNH": mp_lam_upper,
                "masterSPNH": mp_sph_upper,
                "masterSPSH": mp_sph_lower,
                "numEbins": len(energies),
                "numset": 1,
            }
        },
        "NMLparameters": {
            "TKDMasterNameList": {"dmin": 0.05, "npx": npx},
            "MCCLfoilNameList": {
                "Ebinsize": energies[1] - energies[0],
                "Ehistmin": np.min(energies),
                "EkeV": np.max(energies),
                "MCmode": "CSDA",
                "dataname": "crystal_data/al2cu/al2cu_mc_mp_20kv.h5",
                "depthmax": 100.0,
                "depthstep": 1.0,
                "numsx": npx,
                "sig": -20.0,
                "totnum_el": 2000000000,
            },
            "BetheList": {"c1": 4.0, "c2": 8.0, "c3": 50.0, "sgdbdiff": 1.0},
        },
        "EMheader": {
            "TKDmaster": {"ProgramName": np.array([b"EMTKDmaster.f90"], dtype="S15")},
        },
    }

    _dict2hdf5group(dictionary=data, group=f["/"])

    # One chunked data set
    f["EMData/TKDmaster"].create_dataset("mLPSH", data=mp_lam_lower, chunks=True)

    # One byte string with latin-1 stuff
    creation_time = b"12:30:13.559 PM\xf0\x14\x1e\xc8\xbcU"
    f["CrystalData"].create_dataset("CreationTime", data=creation_time)

    f.close()

    yield fpath


# -------------------------- NORDIF formats -------------------------- #


@pytest.fixture
def nordif_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "nordif"


@pytest.fixture
def nordif_renamed_calibration_pattern(
    nordif_path: Path,
) -> Generator[Path, None, None]:
    fname = "Background calibration pattern.bmp"
    f1 = nordif_path / fname
    f2 = f1.rename(f1.with_suffix(".bak"))
    yield f2
    f2.rename(f1)


# -------------------------- Bruker formats -------------------------- #


@pytest.fixture
def bruker_path() -> Generator[Path, None, None]:
    yield DATA_PATH / "bruker_h5ebsd"


@pytest.fixture
def bruker_h5ebsd_file(tmpdir) -> Generator[Path, None, None]:
    """Bruker h5ebsd file with no region of interest."""
    fpath = tmpdir / "patterns.h5"
    create_dummy_bruker_h5ebsd_file(fpath)
    yield fpath


@pytest.fixture
def bruker_h5ebsd_roi_file(tmpdir) -> Generator[Path, None, None]:
    """Bruker h5ebsd file with rectangular region of interest (and SEM
    group under EBSD group).
    """
    path = tmpdir / "patterns_roi.h5"
    create_dummy_bruker_h5ebsd_roi_file(path)
    yield path


@pytest.fixture
def bruker_h5ebsd_nonrectangular_roi_file(tmpdir) -> Generator[Path, None, None]:
    """Bruker h5ebsd file with non-rectangular region of interest (and
    SEM group under EBSD group).
    """
    path = tmpdir / "patterns_roi_nonrectangular.h5"
    create_dummy_bruker_h5ebsd_nonrectangular_roi_file(path)
    yield path

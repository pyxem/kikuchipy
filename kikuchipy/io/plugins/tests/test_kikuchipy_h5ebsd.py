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

import os

import dask.array as da
from hyperspy.api import load as hs_load
from h5py import File, Dataset
import numpy as np
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy.data import nickel_ebsd_small
from kikuchipy.conftest import assert_dictionary
from kikuchipy.io._io import load
from kikuchipy.io.plugins._h5ebsd import _dict2hdf5group
from kikuchipy.io.plugins.kikuchipy_h5ebsd import (
    KikuchipyH5EBSDReader,
    KikuchipyH5EBSDWriter,
)
from kikuchipy.signals.ebsd import EBSD


DIR_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIR_PATH, "../../../data")
KIKUCHIPY_FILE = os.path.join(DATA_PATH, "kikuchipy_h5ebsd/patterns.h5")
KIKUCHIPY_FILE_NO_CHUNKS = os.path.join(
    DATA_PATH, "kikuchipy_h5ebsd/patterns_nochunks.h5"
)
KIKUCHIPY_FILE_GROUP_NAMES = ["My awes0m4 Xcan #! with a long title", "Scan 2"]
BG_FILE = os.path.join(DATA_PATH, "nordif/Background acquisition image.bmp")


class TestH5EBSD:
    def test_repr(self):
        reader = KikuchipyH5EBSDReader(KIKUCHIPY_FILE)
        repr_str_list = repr(reader).split(" ")
        assert repr_str_list[:2] == ["KikuchipyH5EBSDReader", "(0.1):"]
        assert repr_str_list[2][-11:] == "patterns.h5"

    def test_check_file_invalid_version(self, save_path_hdf5):
        f = File(save_path_hdf5, mode="w")
        _dict2hdf5group({"manufacturer": "kikuchipy", "versionn": "0.1"}, f["/"])
        f.close()
        with pytest.raises(IOError, match="(.*) as manufacturer"):
            _ = KikuchipyH5EBSDReader(save_path_hdf5)

    def test_check_file_no_scan_groups(self, save_path_hdf5):
        f = File(save_path_hdf5, mode="w")
        _dict2hdf5group({"manufacturer": "kikuchipy", "version": "0.1"}, f["/"])
        f.close()
        with pytest.raises(IOError, match="(.*) as no top groups"):
            _ = KikuchipyH5EBSDReader(save_path_hdf5)

    def test_dict2hdf5roup(self, save_path_hdf5):
        with File(save_path_hdf5, mode="w") as f:
            with pytest.warns(UserWarning, match="(c, set())"):
                _dict2hdf5group({"a": [np.array(24.5)], "c": set()}, f["/"])


class TestKikuchipyH5EBSD:
    def test_load(self, ni_small_axes_manager):
        s = load(KIKUCHIPY_FILE)

        assert s.data.shape == (3, 3, 60, 60)
        assert_dictionary(s.axes_manager.as_dictionary(), ni_small_axes_manager)

    def test_save_load_xmap(self, detector, save_path_hdf5):
        mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
        rot = Rotation.from_euler(np.deg2rad([[0, 0, 0], [45, 0, 0]]))
        sim1 = mp.get_patterns(
            rotations=rot,
            detector=detector,
            energy=20,
            dtype_out=np.uint8,
            compute=True,
        )
        xmap1 = sim1.xmap.deepcopy()
        assert xmap1.size == 2
        assert np.allclose(xmap1.rotations.data, rot.data)
        pg = xmap1.phases[0].point_group.name
        assert pg == mp.phase.point_group.name
        sim1.save(save_path_hdf5)

        sim2 = kp.load(save_path_hdf5)
        xmap2 = sim2.xmap.deepcopy()
        assert xmap2.size == xmap1.size
        assert xmap2.phases[0].point_group.name == pg

    def test_load_manufacturer(self, save_path_hdf5):
        s = EBSD((255 * np.random.rand(10, 3, 5, 5)).astype(np.uint8))
        s.save(save_path_hdf5)

        # Change manufacturer
        with File(save_path_hdf5, mode="r+") as f:
            manufacturer = f["manufacturer"]
            manufacturer[()] = "Nope".encode()

        with pytest.raises(
            OSError,
            match="(.*) is not a supported h5ebsd file, as 'nope' is not among ",
        ):
            _ = load(save_path_hdf5)

    def test_read_patterns(self, save_path_hdf5):
        s = EBSD((255 * np.random.rand(10, 3, 5, 5)).astype(np.uint8))
        s.save(save_path_hdf5)
        with File(save_path_hdf5, mode="r+") as f:
            del f["Scan 1/EBSD/Data/patterns"]
            with pytest.raises(KeyError, match="Could not find patterns"):
                _ = load(save_path_hdf5)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_load_with_padding(self, save_path_hdf5, lazy, ni_small_axes_manager):
        s = load(KIKUCHIPY_FILE)
        s.save(save_path_hdf5)

        new_n_columns = 4
        with File(save_path_hdf5, mode="r+") as f:
            f["Scan 1/EBSD/Header/n_columns"][()] = new_n_columns
        with pytest.warns(UserWarning, match="Will attempt to load by zero"):
            s_reload = load(save_path_hdf5, lazy=lazy)
        ni_small_axes_manager["axis-1"]["size"] = new_n_columns
        assert_dictionary(s_reload.axes_manager.as_dictionary(), ni_small_axes_manager)

    def test_load_save_cycle(self, save_path_hdf5):
        s = load(KIKUCHIPY_FILE)

        # Check that metadata is read correctly
        assert s.detector.binning == 8
        assert s.metadata.General.title == "patterns My awes0m4 ..."

        s.save(save_path_hdf5, overwrite=True)
        s_reload = load(save_path_hdf5)
        np.testing.assert_equal(s.data, s_reload.data)

        # Change data set name and original filename to make metadata
        # equal
        md = s.metadata
        md2 = s_reload.metadata
        md2.General.title = md.General.title
        md2.General.original_filename = md.General.original_filename
        np.testing.assert_equal(md2.as_dictionary(), md.as_dictionary())

    def test_load_save_hyperspy_cycle(self, tmp_path):
        s = load(KIKUCHIPY_FILE)

        # Perform decomposition to tests if learning results are
        # maintained after saving, reloading and using set_signal_type
        s.change_dtype(np.float32)
        s.decomposition()

        # Write both patterns and learning results to the HSpy file
        # format
        os.chdir(tmp_path)
        file = "patterns.hspy"
        s.save(file)

        # Reload data and use HyperSpy's set_signal_type function
        s_reload = hs_load(file)
        s_reload.set_signal_type("EBSD")

        # Check signal type, patterns and learning results
        assert isinstance(s_reload, EBSD)
        assert np.allclose(s.data, s_reload.data)
        assert np.allclose(
            s.learning_results.factors, s_reload.learning_results.factors
        )

    @pytest.mark.parametrize(
        "scan_group_names",
        (
            KIKUCHIPY_FILE_GROUP_NAMES,
            KIKUCHIPY_FILE_GROUP_NAMES + ["Scan 3"],
            ["Scan 3"],
            KIKUCHIPY_FILE_GROUP_NAMES[1],
        ),
    )
    def test_load_multiple(self, scan_group_names):
        if scan_group_names == KIKUCHIPY_FILE_GROUP_NAMES + ["Scan 3"]:
            with pytest.warns(UserWarning, match="Scan 'Scan 3' is not among "):
                s1, s2 = load(KIKUCHIPY_FILE, scan_group_names=scan_group_names)
        elif scan_group_names == ["Scan 3"]:
            with pytest.raises(OSError, match="Scan 'Scan 3' is not among the"):
                _ = load(KIKUCHIPY_FILE, scan_group_names=scan_group_names)
            return 0
        elif scan_group_names == KIKUCHIPY_FILE_GROUP_NAMES:
            s1, s2 = load(KIKUCHIPY_FILE, scan_group_names=KIKUCHIPY_FILE_GROUP_NAMES)
        else:  # scan_group_names == "Scan 2"
            s2 = load(KIKUCHIPY_FILE, scan_group_names=scan_group_names)
            assert s2.metadata.General.title == "patterns Scan 2"
            s1 = load(KIKUCHIPY_FILE)

        assert np.allclose(s1.data, s2.data)
        with pytest.raises(
            AssertionError,
            match="\nItems are not equal:\nkey='title'\nkey='General'\n\n "
            "ACTUAL: 'patterns My awes0m4 ...'\n DESIRED: 'patterns Scan 2'",
        ):
            np.testing.assert_equal(
                s1.metadata.as_dictionary(), s2.metadata.as_dictionary()
            )
        s2.metadata.General.title = s1.metadata.General.title
        np.testing.assert_equal(
            s1.metadata.as_dictionary(), s2.metadata.as_dictionary()
        )

    def test_load_save_lazy(self, save_path_hdf5):
        s = load(KIKUCHIPY_FILE, lazy=True)
        assert isinstance(s.data, da.Array)
        s.save(save_path_hdf5)
        s_reload = load(save_path_hdf5, lazy=True)
        assert s.data.shape == s_reload.data.shape
        with pytest.raises(OSError, match="Cannot write to an already open"):
            s_reload.save(save_path_hdf5, add_scan=True, scan_number=2)

    def test_load_readonly(self):
        s = load(KIKUCHIPY_FILE, lazy=True)
        keys = ["array-original", "original-array"]
        k = next(
            filter(
                lambda x: isinstance(x, str) and any([x.startswith(j) for j in keys]),
                s.data.dask.keys(),
            )
        )
        mm = s.data.dask[k]
        assert isinstance(mm, Dataset)

    def test_save_fresh(self, save_path_hdf5, tmp_path):
        scan_size = (10, 3)
        pattern_size = (5, 5)
        data_shape = scan_size + pattern_size
        s = EBSD((255 * np.random.rand(*data_shape)).astype(np.uint8))
        s.save(save_path_hdf5, overwrite=True)
        s_reload = load(save_path_hdf5)
        np.testing.assert_equal(s.data, s_reload.data)

        # Test writing of signal to file when no file name is passed to save()
        del s.tmp_parameters.filename
        with pytest.raises(ValueError, match="Filename not defined"):
            s.save(overwrite=True)

        s.metadata.General.original_filename = "an_original_filename"
        os.chdir(tmp_path)
        s.save(overwrite=True)

    @pytest.mark.parametrize("scan_number", (1, 2))
    def test_save_multiple(self, save_path_hdf5, scan_number):
        s1, s2 = load(KIKUCHIPY_FILE, scan_group_names=KIKUCHIPY_FILE_GROUP_NAMES)
        s1.save(save_path_hdf5)
        error = "Invalid scan number"
        with pytest.raises(OSError, match=error), pytest.warns(UserWarning):
            s2.save(save_path_hdf5, add_scan=True)
        if scan_number == 1:
            with pytest.raises(OSError, match=error), pytest.warns(UserWarning):
                s2.save(save_path_hdf5, add_scan=True, scan_number=scan_number)
        else:
            s2.save(save_path_hdf5, add_scan=True, scan_number=scan_number)

    def test_read_lazily_no_chunks(self):
        # First, make sure the data image dataset is not actually chunked
        f = File(KIKUCHIPY_FILE_NO_CHUNKS)
        data_dset = f["Scan 1/EBSD/Data/patterns"]
        assert data_dset.chunks is None
        f.close()

        # Then, make sure it can be read correctly
        s = load(KIKUCHIPY_FILE_NO_CHUNKS, lazy=True)
        assert s.data.chunks == ((60,), (60,))

    def test_save_load_1d_nav(self, save_path_hdf5):
        """Save-load cycle of signals with one navigation dimension."""
        desired_shape = (3, 60, 60)
        desired_nav_extent = (0, 3)
        s = nickel_ebsd_small()

        # One column of patterns
        s_y_only = s.inav[0]
        s_y_only.save(save_path_hdf5)
        s_y_only2 = load(save_path_hdf5)
        assert s_y_only2.data.shape == desired_shape
        assert s_y_only2.axes_manager.navigation_axes[0].name == "y"
        assert s_y_only2.axes_manager.navigation_extent == desired_nav_extent

        # One row of patterns
        s_x_only = s.inav[:, 0]
        s_x_only.save(save_path_hdf5, overwrite=True)
        s_x_only2 = load(save_path_hdf5)
        assert s_x_only2.data.shape == desired_shape
        assert s_x_only2.axes_manager.navigation_axes[0].name == "x"
        assert s_x_only2.axes_manager.navigation_extent == desired_nav_extent

        # Maintain axis name
        s_y_only2.axes_manager["y"].name = "x"
        with pytest.warns(UserWarning, match="^The `xmap`"):
            s_y_only2.save(save_path_hdf5, overwrite=True)
        s_x_only3 = load(save_path_hdf5)
        assert s_x_only3.data.shape == desired_shape
        assert s_x_only3.axes_manager.navigation_axes[0].name == "x"
        assert s_x_only3.axes_manager.navigation_extent == desired_nav_extent

    def test_save_load_0d_nav(self, save_path_hdf5):
        """Save-load cycle of a signal with no navigation dimension."""
        s = nickel_ebsd_small()
        s0 = s.inav[0, 0]
        s0.save(save_path_hdf5)
        with pytest.warns(DeprecationWarning, match="Calling nonzero"):
            s1 = load(save_path_hdf5)
        assert s1.data.shape == (60, 60)
        assert s1.axes_manager.navigation_axes == ()

    def test_save_load_non_square_patterns(self, save_path_hdf5):
        """Ensure non-square patterns are written to file correctly."""
        data_shape = (3, 4, 5, 6)
        data = np.random.randint(
            low=0, high=256, size=np.prod(data_shape), dtype=np.uint8
        ).reshape(data_shape)
        s = EBSD(data)
        s.save(save_path_hdf5)
        s2 = load(save_path_hdf5)
        assert s.data.shape == s2.data.shape
        assert np.allclose(s.data, s2.data)

    def test_load_with_detector_multiple_pc(self, ni_kikuchipy_h5ebsd_file):
        s = kp.load(ni_kikuchipy_h5ebsd_file)
        assert s.detector.pc.shape == (3, 3, 3)

    def test_writer_check_file(self, save_path_hdf5):
        s = kp.data.nickel_ebsd_small(lazy=True)
        f = File(save_path_hdf5, mode="w")
        _dict2hdf5group({"manufacturer": "kikuchipy", "version": "0.1"}, f["/"])
        f.close()
        with pytest.raises(IOError, match="(.*) as no top groups"):
            _ = KikuchipyH5EBSDWriter(save_path_hdf5, s, add_scan=True)

    def test_writer_repr(self, save_path_hdf5):
        s = kp.data.nickel_ebsd_small()
        writer = KikuchipyH5EBSDWriter(save_path_hdf5, s)
        repr_str_list = repr(writer).split(" ")
        assert repr_str_list[0] == "KikuchipyH5EBSDWriter:"
        assert repr_str_list[1][-11:] == "patterns.h5"

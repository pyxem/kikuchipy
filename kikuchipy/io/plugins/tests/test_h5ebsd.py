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

import os

import dask.array as da
from hyperspy.api import load as hs_load
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.exceptions import VisibleDeprecationWarning
from h5py import File, Dataset
import numpy as np
import pytest

from kikuchipy.data import nickel_ebsd_small
from kikuchipy.io._io import load
from kikuchipy.io.plugins.h5ebsd import (
    check_h5ebsd,
    dict2h5ebsdgroup,
    hdf5group2dict,
)
from kikuchipy.signals.ebsd import EBSD
from kikuchipy.signals.util._metadata import metadata_nodes


DIR_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIR_PATH, "../../../data")
KIKUCHIPY_FILE = os.path.join(DATA_PATH, "kikuchipy/patterns.h5")
KIKUCHIPY_FILE_NO_CHUNKS = os.path.join(
    DATA_PATH, "kikuchipy/patterns_nochunks.h5"
)
KIKUCHIPY_FILE_GROUP_NAMES = [
    "My awes0m4 Xcan #! with a long title",
    "Scan 2",
]
EDAX_FILE = os.path.join(DATA_PATH, "edax/patterns.h5")
BRUKER_FILE = os.path.join(DATA_PATH, "bruker/patterns.h5")
BG_FILE = os.path.join(DATA_PATH, "nordif/Background acquisition image.bmp")
AXES_MANAGER = {
    "axis-0": {
        "name": "y",
        "scale": 1.5,
        "offset": 0.0,
        "size": 3,
        "units": "um",
        "navigate": True,
    },
    "axis-1": {
        "name": "x",
        "scale": 1.5,
        "offset": 0.0,
        "size": 3,
        "units": "um",
        "navigate": True,
    },
    "axis-2": {
        "name": "dy",
        "scale": 1.0,
        "offset": 0.0,
        "size": 60,
        "units": "um",
        "navigate": False,
    },
    "axis-3": {
        "name": "dx",
        "scale": 1.0,
        "offset": 0.0,
        "size": 60,
        "units": "um",
        "navigate": False,
    },
}


class Testh5ebsd:
    def test_load_kikuchipy(self):
        s = load(KIKUCHIPY_FILE)

        assert s.data.shape == (3, 3, 60, 60)
        assert s.axes_manager.as_dictionary() == AXES_MANAGER

    def test_load_edax(self):
        with File(EDAX_FILE, mode="r+") as f:
            grid = f["Scan 1/EBSD/Header/Grid Type"]
            grid[()] = "HexGrid".encode()
        with pytest.raises(IOError, match="Only square grids are"):
            _ = load(EDAX_FILE)
        with File(EDAX_FILE, mode="r+") as f:
            grid = f["Scan 1/EBSD/Header/Grid Type"]
            grid[()] = "SqrGrid".encode()

        s = load(EDAX_FILE)
        assert s.data.shape == (3, 3, 60, 60)
        assert s.axes_manager.as_dictionary() == AXES_MANAGER

    def test_load_bruker(self):
        with File(BRUKER_FILE, mode="r+") as f:
            grid = f["Scan 0/EBSD/Header/Grid Type"]
            grid[()] = "hexagonal".encode()
        with pytest.raises(IOError, match="Only square grids are"):
            _ = load(BRUKER_FILE)
        with File(BRUKER_FILE, mode="r+") as f:
            grid = f["Scan 0/EBSD/Header/Grid Type"]
            grid[()] = "isometric".encode()

        s = load(BRUKER_FILE)
        assert s.data.shape == (3, 3, 60, 60)
        assert s.axes_manager.as_dictionary() == AXES_MANAGER

    def test_load_manufacturer(self, save_path_hdf5):
        s = EBSD((255 * np.random.rand(10, 3, 5, 5)).astype(np.uint8))
        s.save(save_path_hdf5)

        # Change manufacturer
        with File(save_path_hdf5, mode="r+") as f:
            manufacturer = f["manufacturer"]
            manufacturer[()] = "Nope".encode()

        with pytest.raises(
            OSError,
            match="Manufacturer Nope not among recognised manufacturers",
        ):
            _ = load(save_path_hdf5)

    @pytest.mark.parametrize(
        "delete, error",
        [
            ("man_ver", ".* is not an h5ebsd file, as manufacturer"),
            ("scans", ".* is not an h5ebsd file, as no top groups with "),
        ],
    )
    def test_check_h5ebsd(self, save_path_hdf5, delete, error):
        s = EBSD((255 * np.random.rand(10, 3, 5, 5)).astype(np.uint8))
        s.save(save_path_hdf5)

        with File(save_path_hdf5, mode="r+") as f:
            if delete == "man_ver":
                del f["manufacturer"]
                del f["version"]
                with pytest.raises(OSError, match=error):
                    check_h5ebsd(f)
            else:
                del f["Scan 1"]
                with pytest.raises(OSError, match=error):
                    check_h5ebsd(f)

    def test_read_patterns(self, save_path_hdf5):
        s = EBSD((255 * np.random.rand(10, 3, 5, 5)).astype(np.uint8))
        s.save(save_path_hdf5)
        with File(save_path_hdf5, mode="r+") as f:
            del f["Scan 1/EBSD/Data/patterns"]
            with pytest.raises(KeyError, match="Could not find patterns"):
                _ = load(save_path_hdf5)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_load_with_padding(self, save_path_hdf5, lazy):
        s = load(KIKUCHIPY_FILE)
        s.save(save_path_hdf5)

        new_n_columns = 4
        with File(save_path_hdf5, mode="r+") as f:
            f["Scan 1/EBSD/Header/n_columns"][()] = new_n_columns
        with pytest.warns(UserWarning, match="Will attempt to load by zero"):
            s_reload = load(save_path_hdf5, lazy=lazy)
        AXES_MANAGER["axis-1"]["size"] = new_n_columns
        assert s_reload.axes_manager.as_dictionary() == AXES_MANAGER

    @pytest.mark.parametrize("remove_phases", (True, False))
    def test_load_save_cycle(self, save_path_hdf5, remove_phases):
        s = load(KIKUCHIPY_FILE)

        # Check that metadata is read correctly
        assert s.metadata.Acquisition_instrument.SEM.Detector.EBSD.xpc == -5.64
        assert s.metadata.General.title == "patterns My awes0m4 ..."

        if remove_phases:
            del s.metadata.Sample.Phases
        s.save(save_path_hdf5, overwrite=True)
        s_reload = load(save_path_hdf5)
        np.testing.assert_equal(s.data, s_reload.data)

        # Change data set name and package version to make metadata equal, and
        # redo deleting of phases
        s_reload.metadata.General.title = s.metadata.General.title
        ebsd_node = metadata_nodes("ebsd")
        s_reload.metadata.set_item(
            ebsd_node + ".version", s.metadata.get_item(ebsd_node + ".version")
        )
        if remove_phases:
            s.metadata.Sample.set_item(
                "Phases", s_reload.metadata.Sample.Phases
            )
        np.testing.assert_equal(
            s_reload.metadata.as_dictionary(), s.metadata.as_dictionary()
        )

    def test_load_save_hyperspy_cycle(self, tmp_path):
        s = load(KIKUCHIPY_FILE)

        # Perform decomposition to tests if learning results are
        # maintained after saving, reloading and using set_signal_type
        s.change_dtype(np.float32)
        s.decomposition()

        # Write both patterns and learning results to the HSpy file
        # format
        #        file = tmp_path / "patterns.hspy"
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
            s1, s2 = load(
                KIKUCHIPY_FILE, scan_group_names=KIKUCHIPY_FILE_GROUP_NAMES
            )
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
        k = next(
            filter(
                lambda x: isinstance(x, str) and x.startswith("array-original"),
                s.data.dask.keys(),
            )
        )
        mm = s.data.dask[k]
        assert isinstance(mm, Dataset)
        with pytest.raises(NotImplementedError):
            s.data[:] = 23

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
        s1, s2 = load(
            KIKUCHIPY_FILE, scan_group_names=KIKUCHIPY_FILE_GROUP_NAMES
        )
        s1.save(save_path_hdf5)
        error = "Invalid scan number"
        with pytest.raises(OSError, match=error), pytest.warns(UserWarning):
            s2.save(save_path_hdf5, add_scan=True)
        if scan_number == 1:
            with pytest.raises(OSError, match=error), pytest.warns(UserWarning):
                s2.save(save_path_hdf5, add_scan=True, scan_number=scan_number)
        else:
            s2.save(save_path_hdf5, add_scan=True, scan_number=scan_number)

    def test_save_edax(self):
        s = load(EDAX_FILE)
        with pytest.raises(OSError, match="Only writing to kikuchipy's"):
            s.save(EDAX_FILE, add_scan=True)

    def test_dict2h5ebsdgroup(self, save_path_hdf5):
        dictionary = {
            "a": [np.array(24.5)],
            "b": DictionaryTreeBrowser(),
            "c": set(),
        }
        with File(save_path_hdf5, mode="w") as f:
            group = f.create_group(name="a_group")
            with pytest.warns(UserWarning, match="The hdf5 writer could not"):
                dict2h5ebsdgroup(dictionary, group)

    def test_read_lazily_no_chunks(self):
        # First, make sure the data image dataset is not actually chunked
        f = File(KIKUCHIPY_FILE_NO_CHUNKS, mode="r")
        data_dset = f["Scan 1/EBSD/Data/patterns"]
        assert data_dset.chunks is None
        f.close()

        # Then, make sure it can be read correctly
        s = load(KIKUCHIPY_FILE_NO_CHUNKS, lazy=True)
        assert s.data.chunks == ((60,), (60,))

    def test_hdf5group2dict_raises_deprecation_warning(self):
        f = File(KIKUCHIPY_FILE, mode="r")
        with pytest.warns(VisibleDeprecationWarning, match="The 'lazy' "):
            _ = hdf5group2dict(group=f["/"], lazy=True)

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
        s1 = load(save_path_hdf5)
        assert s1.data.shape == (60, 60)
        assert s1.axes_manager.navigation_axes == ()

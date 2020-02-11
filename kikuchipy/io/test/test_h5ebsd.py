# -*- coding: utf-8 -*-
# Copyright 2019-2020 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import os

import dask.array as da
import hyperspy.api as hs
from hyperspy.misc.utils import DictionaryTreeBrowser
import h5py
import numpy as np
import pytest

from kikuchipy.io.plugins.h5ebsd import check_h5ebsd, dict2h5ebsdgroup
import kikuchipy as kp

DIR_PATH = os.path.dirname(__file__)
KIKUCHIPY_FILE = os.path.join(DIR_PATH, "../../data/kikuchipy/patterns.h5")
KIKUCHIPY_FILE_NO_CHUNKS = os.path.join(
    DIR_PATH, "../../data/kikuchipy/patterns_nochunks.h5"
)
EDAX_FILE = os.path.join(DIR_PATH, "../../data/edax/patterns.h5")
BRUKER_FILE = os.path.join(DIR_PATH, "../../data/bruker/patterns.h5")
BG_FILE = os.path.join(
    DIR_PATH, "../../data/nordif/Background acquisition pattern.bmp"
)
AXES_MANAGER = {
    "axis-0": {
        "name": "y",
        "scale": 1.5,
        "offset": 0.0,
        "size": 3,
        "units": "μm",
        "navigate": True,
    },
    "axis-1": {
        "name": "x",
        "scale": 1.5,
        "offset": 0.0,
        "size": 3,
        "units": "μm",
        "navigate": True,
    },
    "axis-2": {
        "name": "dy",
        "scale": 1.0,
        "offset": 0.0,
        "size": 60,
        "units": "μm",
        "navigate": False,
    },
    "axis-3": {
        "name": "dx",
        "scale": 1.0,
        "offset": 0.0,
        "size": 60,
        "units": "μm",
        "navigate": False,
    },
}


class Testh5ebsd:
    def test_load_kikuchipy(self):
        s = kp.load(KIKUCHIPY_FILE)

        assert s.data.shape == (3, 3, 60, 60)
        assert s.axes_manager.as_dictionary() == AXES_MANAGER

    def test_load_edax(self):
        with h5py.File(EDAX_FILE, mode="r+") as f:
            grid = f["Scan 1/EBSD/Header/Grid Type"]
            grid[()] = "HexGrid".encode()
        with pytest.raises(IOError, match="Only square grids are"):
            kp.load(EDAX_FILE)
        with h5py.File(EDAX_FILE, mode="r+") as f:
            grid = f["Scan 1/EBSD/Header/Grid Type"]
            grid[()] = "SqrGrid".encode()

        s = kp.load(EDAX_FILE)
        assert s.data.shape == (3, 3, 60, 60)
        assert s.axes_manager.as_dictionary() == AXES_MANAGER

    def test_load_bruker(self):
        with h5py.File(BRUKER_FILE, mode="r+") as f:
            grid = f["Scan 0/EBSD/Header/Grid Type"]
            grid[()] = "hexagonal".encode()
        with pytest.raises(IOError, match="Only square grids are"):
            kp.load(BRUKER_FILE)
        with h5py.File(BRUKER_FILE, mode="r+") as f:
            grid = f["Scan 0/EBSD/Header/Grid Type"]
            grid[()] = "isometric".encode()

        s = kp.load(BRUKER_FILE)
        assert s.data.shape == (3, 3, 60, 60)
        assert s.axes_manager.as_dictionary() == AXES_MANAGER

    def test_load_manufacturer(self, save_path_h5ebsd):
        s = kp.signals.EBSD(
            (255 * np.random.rand(10, 3, 5, 5)).astype(np.uint8)
        )
        s.save(save_path_h5ebsd)

        # Change manufacturer
        with h5py.File(save_path_h5ebsd, mode="r+") as f:
            manufacturer = f["manufacturer"]
            manufacturer[()] = "Nope".encode()

        with pytest.raises(
            OSError,
            match="Manufacturer Nope not among recognised manufacturers",
        ):
            kp.load(save_path_h5ebsd)

    @pytest.mark.parametrize(
        "delete, error",
        [
            ("man_ver", "not an h5ebsd file, as manufacturer"),
            ("scans", "not an h5ebsd file, as no scans"),
        ],
    )
    def test_check_h5ebsd(self, save_path_h5ebsd, delete, error):
        s = kp.signals.EBSD(
            (255 * np.random.rand(10, 3, 5, 5)).astype(np.uint8)
        )
        s.save(save_path_h5ebsd)

        with h5py.File(save_path_h5ebsd, mode="r+") as f:
            if delete == "man_ver":
                del f["manufacturer"]
                del f["version"]
                with pytest.raises(OSError, match=error):
                    check_h5ebsd(f)
            else:
                del f["Scan 1"]
                with pytest.raises(OSError, match=error):
                    check_h5ebsd(f)

    def test_read_patterns(self, save_path_h5ebsd):
        s = kp.signals.EBSD(
            (255 * np.random.rand(10, 3, 5, 5)).astype(np.uint8)
        )
        s.save(save_path_h5ebsd)
        with h5py.File(save_path_h5ebsd, mode="r+") as f:
            del f["Scan 1/EBSD/Data/patterns"]
            with pytest.raises(KeyError, match="Could not find patterns"):
                kp.load(save_path_h5ebsd)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_load_with_padding(self, save_path_h5ebsd, lazy):
        s = kp.load(KIKUCHIPY_FILE)
        s.save(save_path_h5ebsd)

        new_n_columns = 4
        with h5py.File(save_path_h5ebsd, mode="r+") as f:
            f["Scan 1/EBSD/Header/n_columns"][()] = new_n_columns
        with pytest.warns(UserWarning, match="Will attempt to load by zero"):
            s_reload = kp.load(save_path_h5ebsd, lazy=lazy)
        AXES_MANAGER["axis-1"]["size"] = new_n_columns
        assert s_reload.axes_manager.as_dictionary() == AXES_MANAGER

    @pytest.mark.parametrize("remove_phases", (True, False))
    def test_load_save_cycle(self, save_path_h5ebsd, remove_phases):
        s = kp.load(KIKUCHIPY_FILE)

        # Check that metadata is read correctly
        assert s.metadata.Acquisition_instrument.SEM.Detector.EBSD.xpc == -5.64
        assert s.metadata.General.title == "patterns Scan 1"

        if remove_phases:
            del s.metadata.Sample.Phases
        s.save(save_path_h5ebsd, overwrite=True)
        s_reload = kp.load(save_path_h5ebsd)
        np.testing.assert_equal(s.data, s_reload.data)

        # Change data set name and package version to make metadata equal, and
        # redo deleting of phases
        s_reload.metadata.General.title = s.metadata.General.title
        ebsd_node = kp.util.io.metadata_nodes(sem=False)
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
        s = kp.load(KIKUCHIPY_FILE)

        # Perform decomposition to test if learning results are maintained after
        # saving, reloading and using set_signal_type
        s.change_dtype(np.float32)
        s.decomposition()

        # Write both patterns and learning results to the HSpy file format
        os.chdir(tmp_path)
        fname = "patterns.hspy"
        s.save(fname)

        # Reload data and use HyperSpy's set_signal_type function
        s_reload = hs.load(fname)
        s_reload.set_signal_type("EBSD")

        # Check signal type, patterns and learning results
        assert isinstance(s_reload, kp.signals.EBSD)
        np.testing.assert_equal(s.data, s_reload.data)
        np.testing.assert_equal(
            s.learning_results.factors, s_reload.learning_results.factors
        )

    @pytest.mark.parametrize("scans", ([1, 2], [1, 2, 3], [3,], 2))
    def test_load_multiple(self, scans):
        if scans == [1, 2, 3]:
            with pytest.warns(UserWarning, match="Scan 3 is not among the"):
                s1, s2 = kp.load(KIKUCHIPY_FILE, scans=scans)
        elif scans == [
            3,
        ]:
            with pytest.raises(OSError, match="Scan 3 is not among the"):
                kp.load(KIKUCHIPY_FILE, scans=scans)
            return 0
        elif scans == [1, 2]:
            s1, s2 = kp.load(KIKUCHIPY_FILE, scans=scans)
        else:  # scans == 2
            s2 = kp.load(KIKUCHIPY_FILE, scans=2)
            assert s2.metadata.General.title == "patterns Scan 2"
            s1 = kp.load(KIKUCHIPY_FILE)

        np.testing.assert_equal(s1.data, s2.data)
        with pytest.raises(
            AssertionError,
            match="\nItems are not equal:\nkey='title'\nkey='General'\n\n "
            "ACTUAL: 'patterns Scan 1'\n DESIRED: 'patterns Scan 2'",
        ):
            np.testing.assert_equal(
                s1.metadata.as_dictionary(), s2.metadata.as_dictionary()
            )
        s2.metadata.General.title = s1.metadata.General.title
        np.testing.assert_equal(
            s1.metadata.as_dictionary(), s2.metadata.as_dictionary()
        )

    def test_load_save_lazy(self, save_path_h5ebsd):
        s = kp.load(KIKUCHIPY_FILE, lazy=True)
        assert isinstance(s.data, da.Array)
        s.save(save_path_h5ebsd)
        s_reload = kp.load(save_path_h5ebsd, lazy=True)
        assert s.data.shape == s_reload.data.shape
        with pytest.raises(OSError, match="Cannot write to an already open"):
            s_reload.save(save_path_h5ebsd, add_scan=True, scan_number=2)

    def test_load_readonly(self):
        s = kp.load(KIKUCHIPY_FILE, lazy=True)
        k = next(
            filter(
                lambda x: isinstance(x, str) and x.startswith("array-original"),
                s.data.dask.keys(),
            )
        )
        mm = s.data.dask[k]
        assert isinstance(mm, h5py.Dataset)
        with pytest.raises(NotImplementedError):
            s.data[:] = 23

    def test_save_fresh(self, save_path_h5ebsd, tmp_path):
        scan_size = (10, 3)
        pattern_size = (5, 5)
        data_shape = scan_size + pattern_size
        s = kp.signals.EBSD(
            (255 * np.random.rand(*data_shape)).astype(np.uint8)
        )
        s.save(save_path_h5ebsd, overwrite=True)
        s_reload = kp.load(save_path_h5ebsd)
        np.testing.assert_equal(s.data, s_reload.data)

        # Test writing of signal to file when no file name is passed to save()
        del s.tmp_parameters.filename
        with pytest.raises(ValueError, match="Filename not defined"):
            s.save(overwrite=True)

        s.metadata.General.original_filename = "an_original_filename"
        os.chdir(tmp_path)
        s.save(overwrite=True)

    @pytest.mark.parametrize("scan_number", (1, 2))
    def test_save_multiple(self, save_path_h5ebsd, scan_number):
        s1, s2 = kp.load(KIKUCHIPY_FILE, scans=[1, 2])
        s1.save(save_path_h5ebsd)
        error = "Invalid scan number"
        with pytest.raises(OSError, match=error), pytest.warns(UserWarning):
            s2.save(save_path_h5ebsd, add_scan=True)
        if scan_number == 1:
            with pytest.raises(OSError, match=error), pytest.warns(UserWarning):
                s2.save(
                    save_path_h5ebsd, add_scan=True, scan_number=scan_number
                )
        else:
            s2.save(save_path_h5ebsd, add_scan=True, scan_number=scan_number)

    def test_save_edax(self):
        s = kp.load(EDAX_FILE)
        with pytest.raises(OSError, match="Only writing to KikuchiPy's"):
            s.save(EDAX_FILE, add_scan=True)

    def test_dict2h5ebsdgroup(self, save_path_h5ebsd):
        dictionary = {
            "a": [np.array(24.5)],
            "b": DictionaryTreeBrowser(),
            "c": set(),
        }
        with h5py.File(save_path_h5ebsd, mode="w") as f:
            group = f.create_group(name="a_group")
            with pytest.warns(UserWarning, match="The hdf5 writer could not"):
                dict2h5ebsdgroup(dictionary, group)

    def test_read_lazily_no_chunks(self):
        # First, make sure the data pattern dataset is not actually chunked
        f = h5py.File(KIKUCHIPY_FILE_NO_CHUNKS, mode="r")
        data_dset = f["Scan 1/EBSD/Data/patterns"]
        assert data_dset.chunks is None
        f.close()

        # Then, make sure it can be read correctly
        s = kp.load(KIKUCHIPY_FILE_NO_CHUNKS, lazy=True)
        assert s.data.chunks == ((60,), (60,))

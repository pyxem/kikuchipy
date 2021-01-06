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
import os
import tempfile

import numpy as np
import pytest

from kikuchipy.io._io import load
from kikuchipy.io.plugins import h5ebsd
from kikuchipy.io._io import _assign_signal_subclass, _dict2signal
from kikuchipy.signals.ebsd import EBSD, LazyEBSD

DIR_PATH = os.path.dirname(__file__)
KIKUCHIPY_FILE = os.path.join(DIR_PATH, "../../data/kikuchipy/patterns.h5")


class TestIO:
    @pytest.mark.parametrize("filename", ("im_not_here.h5", "unsupported.h4"))
    def test_load(self, filename):
        if filename == "im_not_here.h5":
            with pytest.raises(IOError, match="No filename matches"):
                _ = load(filename)
        else:
            s = load(KIKUCHIPY_FILE)
            with tempfile.TemporaryDirectory() as tmp:
                file_path = os.path.join(tmp, "supported.h5")
                s.save(file_path)
                new_file_path = os.path.join(tmp, filename)
                os.rename(file_path, new_file_path)
                with pytest.raises(IOError, match="Could not read"):
                    _ = load(new_file_path)
            gc.collect()

    def test_dict2signal(self):
        scan_dict = h5ebsd.file_reader(KIKUCHIPY_FILE)[0]
        scan_dict["metadata"]["Signal"]["record_by"] = "not-image"
        with pytest.raises(ValueError, match="kikuchipy only supports"):
            _ = _dict2signal(scan_dict)

    @pytest.mark.parametrize(
        "dtype, lazy, signal_dimension, signal_type",
        [
            (np.dtype("complex"), True, 2, "EBSD"),
            (np.dtype("uint8"), False, 2, "EBSD"),
            (np.dtype("uint8"), True, 2, "EBSD"),
            (np.dtype("uint8"), False, -1, "EBSD"),
            (np.dtype("uint8"), False, 2, ""),
        ],
    )
    def test_assign_signal_subclass(
        self, dtype, lazy, signal_dimension, signal_type
    ):
        if "complex" in dtype.name:
            with pytest.raises(ValueError, match="Data type"):
                _ = _assign_signal_subclass(
                    dtype=dtype,
                    signal_dimension=signal_dimension,
                    signal_type=signal_type,
                    lazy=lazy,
                )
        elif not isinstance(signal_dimension, int) or signal_dimension < 0:
            with pytest.raises(ValueError, match="Signal dimension must be"):
                _ = _assign_signal_subclass(
                    dtype=dtype,
                    signal_dimension=signal_dimension,
                    signal_type=signal_type,
                    lazy=lazy,
                )
        elif signal_type == "":
            with pytest.raises(ValueError, match="No kikuchipy signals match"):
                _ = _assign_signal_subclass(
                    dtype=dtype,
                    signal_dimension=signal_dimension,
                    signal_type=signal_type,
                    lazy=lazy,
                )
        else:
            signal = _assign_signal_subclass(
                dtype=dtype,
                signal_dimension=signal_dimension,
                signal_type=signal_type,
                lazy=lazy,
            )
            if not lazy:
                assert signal == EBSD
            else:
                assert signal == LazyEBSD

    @pytest.mark.parametrize("extension", ("", ".h4"))
    def test_save_extensions(self, extension):
        s = load(KIKUCHIPY_FILE)
        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "supported" + extension)
            if extension == "":
                s.save(file_path)
                assert os.path.isfile(file_path + ".h5") is True
            else:  # extension == '.h4'
                with pytest.raises(ValueError, match="'h4' does not"):
                    s.save(file_path)
            gc.collect()

    def test_save_data_dimensions(self):
        s = load(KIKUCHIPY_FILE)
        s.axes_manager.set_signal_dimension(3)
        with pytest.raises(ValueError, match="This file format cannot write"):
            s.save()

    def test_save_to_existing_file(self, save_path_hdf5):
        s = load(KIKUCHIPY_FILE)
        s.save(save_path_hdf5)
        with pytest.warns(UserWarning, match="Your terminal does not"):
            s.save(save_path_hdf5, scan_number=2)
        with pytest.raises(ValueError, match="overwrite parameter can"):
            s.save(
                save_path_hdf5,
                scan_number=2,
                overwrite="False",
                add_scan=False,
            )
        s.save(save_path_hdf5, scan_number=2, overwrite=False, add_scan=False)
        with pytest.raises(OSError, match="Scan 'Scan 2' is not among the"):
            _ = load(save_path_hdf5, scan_group_names="Scan 2")

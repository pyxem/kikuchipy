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

import os

import hyperspy.api as hs
import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.io._io import _assign_signal_subclass, _dict2signal
from kikuchipy.io.plugins.kikuchipy_h5ebsd._api import (
    file_reader as kp_h5ebsd_file_reader,
)


class TestIO:
    def test_load_unsupported_extension(self, kikuchipy_h5ebsd_path, tmpdir):
        s = kp.load(kikuchipy_h5ebsd_path / "patterns.h5")
        file_path = tmpdir / "supported.h5"
        s.save(file_path)
        new_file_path = tmpdir / "unsupported.h4"
        file_path.rename(new_file_path)
        with pytest.raises(IOError, match="Could not read"):
            kp.load(new_file_path)

    def test_load_missing_file(self, kikuchipy_h5ebsd_path, tmpdir):
        with pytest.raises(IOError, match="No filename matches"):
            kp.load("im_not_here.h5")

    def test_dict2signal(self, kikuchipy_h5ebsd_path):
        scan_dict, *_ = kp_h5ebsd_file_reader(kikuchipy_h5ebsd_path / "patterns.h5")
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
    def test_assign_signal_subclass(self, dtype, lazy, signal_dimension, signal_type):
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
                assert signal == kp.signals.EBSD
            else:
                assert signal == kp.signals.LazyEBSD

    @pytest.mark.parametrize("extension", ("", ".h4"))
    def test_save_extensions(self, kikuchipy_h5ebsd_path, extension, tmpdir):
        s = kp.load(kikuchipy_h5ebsd_path / "patterns.h5")
        file_path = tmpdir / ("supported" + extension)
        if extension == "":
            s.save(file_path)
            assert os.path.exists(file_path + ".h5") is True
        else:  # extension == '.h4'
            with pytest.raises(ValueError, match="'h4' does not"):
                s.save(file_path)

    def test_save_data_dimensions(self, tmpdir):
        s = kp.signals.EBSD(np.zeros((2, 3, 4, 5, 6)))
        with pytest.raises(ValueError, match="Chosen IO plugin 'kikuchipy_h5ebsd' "):
            s.save(tmpdir / "test.h5")

    def test_save_to_existing_file(self, save_path_hdf5, kikuchipy_h5ebsd_path):
        s = kp.load(kikuchipy_h5ebsd_path / "patterns.h5")
        s.save(save_path_hdf5)
        with pytest.warns(UserWarning, match="Your terminal does not"):
            s.save(save_path_hdf5, scan_number=2)
        with pytest.raises(ValueError, match="overwrite parameter can"):
            s.save(save_path_hdf5, scan_number=2, overwrite="False", add_scan=False)
        s.save(save_path_hdf5, scan_number=2, overwrite=False, add_scan=False)
        with pytest.raises(OSError, match="Scan 'Scan 2' is not among the"):
            _ = kp.load(save_path_hdf5, scan_group_names="Scan 2")


class TestHSPYWrite:
    def test_hspy_write(self, tmpdir, dummy_signal):
        file_path = str(tmpdir / "test.hspy")
        s = dummy_signal.as_lazy()
        s.save(file_path)
        s2 = hs.load(file_path, signal_type="EBSD")
        assert np.allclose(dummy_signal.data, s2.data)

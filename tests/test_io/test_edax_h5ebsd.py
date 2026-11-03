#
# Copyright 2019-2026 the kikuchipy developers
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
#

import shutil

import h5py
import numpy as np
import pytest

import kikuchipy as kp


class TestEDAXH5EBSD:
    def test_load(
        self, edax_h5ebsd_path, tmp_path, ni_small_axes_manager, assert_dictionary_func
    ):
        file = edax_h5ebsd_path / "patterns.h5"
        s = kp.load(file)
        assert s.data.shape == (3, 3, 60, 60)
        assert_dictionary_func(s.axes_manager.as_dictionary(), ni_small_axes_manager)
        mag = s.metadata.Acquisition_instrument.SEM.magnification

        # No 'SEM-PRIAS' group available
        tmp_file = tmp_path / file.name
        shutil.copy(file, tmp_file)
        with h5py.File(tmp_file, mode="r+") as f:
            f["Scan 1"].move("SEM-PRIAS Images", "SSEM-PRIAS Images")
        s2 = kp.load(tmp_file)
        assert s2.metadata.Acquisition_instrument.SEM.magnification != mag

        # Not a square grid
        with h5py.File(tmp_file, mode="r+") as f:
            grid = f["Scan 1/EBSD/Header/Grid Type"]
            grid[()] = "HexGrid".encode()
        with pytest.raises(IOError, match="Only square grids are"):
            _ = kp.load(tmp_file)

    @pytest.mark.parametrize("azimuthal, desired", [(0, 0), (2.5, -2.5), (-1, 1)])
    def test_load_detector(self, edax_h5ebsd_path, tmp_path, azimuthal, desired):
        """Azimuthal angle is negated to give the same geometry as with
        kikuchipy < 0.12.0.
        """
        file = edax_h5ebsd_path / "patterns.h5"
        tmp_file = tmp_path / file.name
        shutil.copy(file, tmp_file)
        with h5py.File(tmp_file, mode="r+") as f:
            header = f["Scan 1/EBSD/Header"]
            header["Camera Azimuthal Angle"][()] = azimuthal
            header["Camera Elevation Angle"][()] = 4.2

        det = kp.load(tmp_file).detector
        assert np.isclose(det.azimuthal, desired)
        assert np.copysign(1, det.azimuthal) == np.copysign(1, desired)  # Not -0.0
        assert np.isclose(det.tilt, 4.2)
        assert np.isclose(det.sample_tilt, 75.7)

    def test_save_error(self, edax_h5ebsd_path):
        file = edax_h5ebsd_path / "patterns.h5"
        s = kp.load(file)
        with pytest.raises(OSError, match="(.*) is not a supported kikuchipy h5ebsd "):
            s.save(file, add_scan=True)

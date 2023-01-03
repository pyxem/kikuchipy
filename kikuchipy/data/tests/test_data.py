# Copyright 2019-2023 The kikuchipy developers
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
from pathlib import Path

from dask.array import Array
import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.data._data import Dataset, marshall


class TestData:
    def test_load_ni_ebsd_small(self):
        s = kp.data.nickel_ebsd_small()

        assert isinstance(s, kp.signals.EBSD)
        assert s.data.shape == (3, 3, 60, 60)

        s_lazy = kp.data.nickel_ebsd_small(lazy=True)

        assert isinstance(s_lazy, kp.signals.LazyEBSD)
        assert isinstance(s_lazy.data, Array)

        dset = Dataset("kikuchipy_h5ebsd/patterns.h5")
        assert dset.url is None

    def test_load_ni_ebsd_master_pattern_small(self):
        """Can be read."""
        mp = kp.data.nickel_ebsd_master_pattern_small()
        assert mp.data.shape == (401, 401)

    @pytest.mark.parametrize(
        "projection, hemisphere, desired_shape",
        [
            ("lambert", "upper", (401, 401)),
            ("lambert", "both", (2, 401, 401)),
            ("stereographic", "lower", (401, 401)),
            ("stereographic", "both", (2, 401, 401)),
        ],
    )
    def test_load_ni_ebsd_master_pattern_small_kwargs(
        self, projection, hemisphere, desired_shape
    ):
        """Master patterns in both stereographic and Lambert projections
        can be loaded as expected.
        """
        mp = kp.data.nickel_ebsd_master_pattern_small(
            projection=projection, hemisphere=hemisphere
        )

        assert isinstance(mp, kp.signals.EBSDMasterPattern)
        assert mp.data.shape == desired_shape
        assert np.issubdtype(mp.data.dtype, np.uint8)
        assert mp.projection == projection
        assert mp.hemisphere == hemisphere

        mp_lazy = kp.data.nickel_ebsd_master_pattern_small(lazy=True)

        assert isinstance(mp_lazy, kp.signals.LazyEBSDMasterPattern)
        assert isinstance(mp_lazy.data, Array)

    def test_not_allow_download_raises(self):
        """Not passing `allow_download` raises expected error."""
        file = Path(marshall.path, "data/nickel_ebsd_large/patterns.h5")

        # Rename file (dangerous!)
        new_name = str(file) + ".bak"
        rename = False
        if file.exists():  # pragma: no cover
            rename = True
            os.rename(file, new_name)

        with pytest.raises(ValueError, match="File data/nickel_ebsd_large/patterns.h5"):
            _ = kp.data.nickel_ebsd_large()

        # Revert rename
        if rename:  # pragma: no cover
            os.rename(new_name, file)

    def test_load_ni_ebsd_large_allow_download(self):
        """Download from external."""
        s = kp.data.nickel_ebsd_large(lazy=True, allow_download=True)

        assert isinstance(s, kp.signals.LazyEBSD)
        assert s.data.shape == (55, 75, 60, 60)
        assert np.issubdtype(s.data.dtype, np.uint8)

    def test_load_si_ebsd_moving_screen_in(self):
        """Download external Si pattern."""
        s = kp.data.silicon_ebsd_moving_screen_in(allow_download=True)

        assert s.data.shape == (480, 480)
        assert s.data.dtype == np.uint8
        assert isinstance(s.static_background, np.ndarray)

    def test_load_si_ebsd_moving_screen_out5mm(self):
        """Download external Si pattern."""
        s = kp.data.silicon_ebsd_moving_screen_out5mm(allow_download=True)

        assert s.data.shape == (480, 480)
        assert s.data.dtype == np.uint8
        assert isinstance(s.static_background, np.ndarray)

    def test_load_si_ebsd_moving_screen_out10mm(self):
        """Download external Si pattern."""
        s = kp.data.silicon_ebsd_moving_screen_out10mm(allow_download=True)

        assert s.data.shape == (480, 480)
        assert s.data.dtype == np.uint8
        assert isinstance(s.static_background, np.ndarray)

    def test_si_wafer(self):
        """Test set up of Si wafer dataset (without downloading)."""
        with pytest.raises(ValueError, match="File data/si_wafer/Pattern.dat must be "):
            _ = kp.data.si_wafer()

        dset = Dataset("si_wafer/Pattern.dat", collection_name="ebsd_si_wafer.zip")
        assert dset.file_path is None
        assert dset.file_path_str is None
        assert not dset.is_in_package
        assert not dset.is_in_cache
        assert dset.is_in_collection
        assert isinstance(dset.file_relpath, Path)
        assert (
            str(dset.file_relpath)
            == dset.file_relpath_str
            == "data/si_wafer/Pattern.dat"
        )
        assert str(dset.file_directory) == "si_wafer"
        assert dset.md5_hash is None

        with pytest.raises(ValueError, match="File data/si_wafer/Pattern.dat must be "):
            _ = dset.fetch_file_path()

    def test_ni_gain0(self):
        """Test set up of polycrystalline recrystallized Ni dataset
        (without downloading).
        """
        with pytest.raises(ValueError, match="File data/ni_gain0/Pattern.dat must be "):
            _ = kp.data.ni_gain0()

        dset = Dataset("ni_gain0/Pattern.dat", collection_name="scan1_gain0db.zip")
        assert (
            str(dset.file_relpath)
            == dset.file_relpath_str
            == "data/ni_gain0/Pattern.dat"
        )
        assert str(dset.file_directory) == "ni_gain0"

        with pytest.raises(ValueError, match="File data/ni_gain0/Pattern.dat must be "):
            _ = dset.fetch_file_path()

    def test_ni_gain0_calibration(self):
        """Test set up of calibration patterns from polycrystalline
        recrystallized Ni dataset (without downloading).
        """
        with pytest.raises(ValueError, match="File data/ni_gain0/Setting.txt must be "):
            _ = kp.data.ni_gain0_calibration()

        dset = Dataset("ni_gain0/Setting.txt", collection_name="scan1_gain0db.zip")
        assert (
            str(dset.file_relpath)
            == dset.file_relpath_str
            == "data/ni_gain0/Setting.txt"
        )
        assert str(dset.file_directory) == "ni_gain0"

        with pytest.raises(ValueError, match="File data/ni_gain0/Setting.txt must be "):
            _ = dset.fetch_file_path()

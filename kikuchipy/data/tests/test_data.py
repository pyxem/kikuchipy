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
from pathlib import Path

from dask.array import Array
import numpy as np
import pytest

from kikuchipy import data
from kikuchipy.signals import EBSD, LazyEBSD, EBSDMasterPattern, LazyEBSDMasterPattern


class TestData:
    def test_load_nickel_ebsd_small(self):
        s = data.nickel_ebsd_small()

        assert isinstance(s, EBSD)
        assert s.data.shape == (3, 3, 60, 60)

        s_lazy = data.nickel_ebsd_small(lazy=True)

        assert isinstance(s_lazy, LazyEBSD)
        assert isinstance(s_lazy.data, Array)

    def test_load_nickel_ebsd_master_pattern_small(self):
        """Can be read."""
        mp = data.nickel_ebsd_master_pattern_small()
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
    def test_load_nickel_ebsd_master_pattern_small_kwargs(
        self, projection, hemisphere, desired_shape
    ):
        """Master patterns in both stereographic and Lambert projections
        can be loaded as expected.
        """
        mp = data.nickel_ebsd_master_pattern_small(
            projection=projection, hemisphere=hemisphere
        )

        assert isinstance(mp, EBSDMasterPattern)
        assert mp.data.shape == desired_shape
        assert np.issubdtype(mp.data.dtype, np.uint8)
        assert mp.projection == projection
        assert mp.hemisphere == hemisphere

        mp_lazy = data.nickel_ebsd_master_pattern_small(lazy=True)

        assert isinstance(mp_lazy, LazyEBSDMasterPattern)
        assert isinstance(mp_lazy.data, Array)

    def test_not_allow_download_raises(self):
        """Not passing `allow_download` raises expected error."""
        file = Path(data._fetcher.path, "data/nickel_ebsd_large/patterns.h5")

        # Rename file (dangerous!)
        new_name = str(file) + ".bak"
        rename = False
        if file.exists():  # pragma: no cover
            rename = True
            os.rename(file, new_name)

        with pytest.raises(ValueError, match="Dataset nickel_ebsd_large/patterns.h5"):
            _ = data.nickel_ebsd_large()

        # Revert rename
        if rename:  # pragma: no cover
            os.rename(new_name, file)

    def test_load_nickel_ebsd_large_allow_download(self):
        """Download from external."""
        s = data.nickel_ebsd_large(lazy=True, allow_download=True)

        assert isinstance(s, LazyEBSD)
        assert s.data.shape == (55, 75, 60, 60)
        assert np.issubdtype(s.data.dtype, np.uint8)

    def test_load_silicon_ebsd_moving_screen_in(self):
        """Download external Si pattern."""
        s = data.silicon_ebsd_moving_screen_in(allow_download=True)

        assert s.data.shape == (480, 480)
        assert s.data.dtype == np.uint8
        assert isinstance(
            s.metadata.Acquisition_instrument.SEM.Detector.EBSD.static_background,
            np.ndarray,
        )

    def test_load_silicon_ebsd_moving_screen_out5mm(self):
        """Download external Si pattern."""
        s = data.silicon_ebsd_moving_screen_out5mm(allow_download=True)

        assert s.data.shape == (480, 480)
        assert s.data.dtype == np.uint8
        assert isinstance(
            s.metadata.Acquisition_instrument.SEM.Detector.EBSD.static_background,
            np.ndarray,
        )

    def test_load_silicon_ebsd_moving_screen_out10mm(self):
        """Download external Si pattern."""
        s = data.silicon_ebsd_moving_screen_out10mm(allow_download=True)

        assert s.data.shape == (480, 480)
        assert s.data.dtype == np.uint8
        assert isinstance(
            s.metadata.Acquisition_instrument.SEM.Detector.EBSD.static_background,
            np.ndarray,
        )

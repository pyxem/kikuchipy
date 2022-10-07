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

import imageio.v3 as iio
import numpy as np
import pytest

import kikuchipy as kp


class TestEBSDDirectory:
    def test_load(self, ebsd_directory):
        """Patterns are read correctly."""
        s_ref = kp.data.nickel_ebsd_small()
        s = kp.load(os.path.join(ebsd_directory, "*.tif"))
        assert isinstance(s, kp.signals.EBSD)
        assert np.allclose(s.data, s_ref.data)

    def test_load_progressbar(self, ebsd_directory, capsys):
        """Asking for a progressbar works."""
        _ = kp.load(os.path.join(ebsd_directory, "*.tif"))
        captured = capsys.readouterr()
        assert captured.out == ""

        _ = kp.load(os.path.join(ebsd_directory, "*.tif"), show_progressbar=True)
        captured = capsys.readouterr()
        assert "100% Completed" in captured.out

    def test_load_lazy(self, ebsd_directory):
        """Lazy loading works."""
        s_lazy = kp.load(os.path.join(ebsd_directory, "*.tif"), lazy=True)
        assert isinstance(s_lazy, kp.signals.LazyEBSD)

    @pytest.mark.parametrize(
        "ebsd_directory, xy_pattern, nav_shape, fname",
        [
            (("_x{}y{}.png", (9, 1)), None, (9, 1), "*.png"),
            (("-{}-{}.tif", (3, 3)), None, (3, 3), "*.tif"),
            (("-x{}-y{}.bmp", (1, 9)), r"-x(\d+)-y(\d+).bmp", (1, 9), "*.bmp"),
        ],
        indirect=["ebsd_directory"],
    )
    def test_xy_pattern(self, ebsd_directory, xy_pattern, nav_shape, fname):
        """Passing various filename patterns work."""
        s = kp.load(os.path.join(ebsd_directory, fname), xy_pattern=xy_pattern)
        assert s.data.shape == nav_shape + (60, 60)

    @pytest.mark.parametrize(
        "ebsd_directory", [("_x{}y{}.tif", (3, 3))], indirect=["ebsd_directory"]
    )
    def test_warns_more_patterns(self, ebsd_directory):
        """Warns when there are more patterns in directory than
        filenames suggest.
        """
        s0 = kp.data.nickel_ebsd_small()
        iio.imwrite(os.path.join(ebsd_directory, "pattern_x10y50.tif"), s0.data[0, 0])

        with pytest.warns(UserWarning, match="Returned signal will have one "):
            s = kp.load(os.path.join(ebsd_directory, "*.tif"))
        assert s.axes_manager.navigation_shape == (10,)

    @pytest.mark.parametrize(
        "ebsd_directory", [("_xx{}yy{}.tif", (3, 3))], indirect=["ebsd_directory"]
    )
    def test_warns_coordinates(self, ebsd_directory):
        """Warns when navigation coordinates could not be read from
        filenames.
        """
        with pytest.warns(UserWarning, match="Returned signal will have one "):
            s = kp.load(os.path.join(ebsd_directory, "*.tif"))
        assert s.axes_manager.navigation_shape == (9,)

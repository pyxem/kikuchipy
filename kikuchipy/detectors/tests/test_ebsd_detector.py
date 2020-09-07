# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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

import numpy as np
import pytest

from kikuchipy.detectors import EBSDDetector


class TestEBSDDetector:
    def test_init(self, pc1):
        """Initialization works."""
        shape = (1, 2)
        px_size = 3
        binning = 4
        tilt = 5
        det = EBSDDetector(
            shape=shape, px_size=px_size, binning=binning, tilt=tilt, pc=pc1,
        )
        assert det.shape == shape
        assert det.aspect_ratio == 0.5

    @pytest.mark.parametrize(
        "nav_shape, desired_nav_shape, desired_nav_dim",
        [
            ((), (1,), 1),
            ((1,), (1,), 1),
            ((10, 1), (10,), 1),
            ((10, 10, 1), (10, 10), 2),
        ],
    )
    def test_nav_shape_dim(
        self, pc1, nav_shape, desired_nav_shape, desired_nav_dim
    ):
        """Navigation shape and dimension is derived correctly from PC shape."""
        det = EBSDDetector(pc=np.tile(pc1, nav_shape))
        assert det.navigation_shape == desired_nav_shape
        assert det.navigation_dimension == desired_nav_dim

    @pytest.mark.parametrize("pc_type", [list, tuple, np.asarray])
    def test_pc_initialization(self, pc1, pc_type):
        """Initialize PC of valid types."""
        det = EBSDDetector(pc=pc_type(pc1))
        assert isinstance(det.pc, np.ndarray)

    @pytest.mark.parametrize(
        (
            "shape, px_size, binning, pc, ssd, width, height, size, "
            "shape_unbinned, px_size_binned"
        ),
        [
            # fmt: off
            (
                (60, 60), 70, 8, [1, 1, 0.5], 16800, 33600, 33600, 3600,
                (480, 480), 560,
            ),
            (
                (60, 60), 70, 8, [1, 1, 0.7], 23520, 33600, 33600, 3600,
                (480, 480), 560,
            ),
            (
                (480, 460), 70, 0.5, [1, 1, 0.7], 11760, 16100, 16800, 220800,
                (240, 230), 35,
            ),
            (
                (340, 680), 40, 2, [1, 1, 0.7], 19040, 54400, 27200, 231200,
                (680, 1360), 80,
            ),
            # fmt: on
        ],
    )
    def test_detector_dimensions(
        self,
        shape,
        px_size,
        binning,
        pc,
        ssd,
        width,
        height,
        size,
        shape_unbinned,
        px_size_binned,
    ):
        """Initialization yields expected derived values."""
        detector = EBSDDetector(
            shape=shape, px_size=px_size, binning=binning, pc=pc
        )
        assert detector.specimen_scintillator_distance == ssd
        assert detector.width == width
        assert detector.height == height
        assert detector.size == size
        assert detector.shape_unbinned == shape_unbinned
        assert detector.px_size_binned == px_size_binned

    @pytest.mark.parametrize(
        "pc, convention, bruker, tsl, oxford, emsoft",
        # fmt: off
        [
            (
                [3.4848, 114.2016, 15767.7], "emsoft",
                [0.50726, 0.26208, 0.55489], [0.50726, 0.73792, 0.55489],
                [0.50726, 0.73792, 0.55489], [3.4848, 114.2016, 15767.7],
            )
        ],
        # fmt: on
    )
    def test_pc_conversions(self, pc, convention, bruker, tsl, oxford, emsoft):
        """Conversions between PC conventions."""
        detector = EBSDDetector(
            shape=(480, 480),
            binning=1,
            px_size=59.2,
            pc=pc,
            convention=convention,
        )
        assert np.allclose(detector.pc, bruker, atol=1e-4)
        assert np.allclose(detector.to_bruker(), bruker, atol=1e-4)
        assert np.allclose(detector.to_emsoft(), emsoft, atol=1e-4)
        assert np.allclose(detector.to_tsl(), tsl, atol=1e-4)

    def test_repr(self):
        """Expected string representation."""
        pass

    def test_deepcopy(self):
        """Yields the expected parameters and an actual deep copy."""
        pass

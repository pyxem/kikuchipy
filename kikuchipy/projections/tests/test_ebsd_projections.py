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

import numpy as np
from orix.quaternion import Rotation
import pytest

from kikuchipy.projections.ebsd_projections import detector2sample


class TestEBSDProjections:
    @pytest.mark.parametrize(
        "convention, desired_rotation",
        [
            (None, np.array([[0, -0.9, 0.5], [1, 0, 0], [0, 0.5, 0.9]])),
            ("tsl", np.array([[0, -0.9, 0.5], [1, 0, 0], [0, 0.5, 0.9]])),
            ("bruker", np.array([[1, 0, 0], [0, 0.9, -0.5], [0, 0.5, 0.9]])),
        ],
    )
    def test_detector2sample(self, convention, desired_rotation):
        r_matrix = detector2sample(
            sample_tilt=70, detector_tilt=10, convention=convention
        )
        assert np.allclose(r_matrix, desired_rotation, atol=0.1)

    def test_rotate_tsl2bruker(self, r_tsl2bruker):
        s_tilt = 70
        d_tilt = 10
        r_tsl_matrix = detector2sample(sample_tilt=s_tilt, detector_tilt=d_tilt)
        r_tsl = Rotation.from_matrix(r_tsl_matrix)
        r_bruker_matrix = detector2sample(
            sample_tilt=s_tilt, detector_tilt=d_tilt, convention="bruker"
        )
        r_bruker = Rotation.from_matrix(r_bruker_matrix)
        assert np.allclose((~r_tsl2bruker * r_tsl).data, r_bruker.data)

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
from orix.quaternion.orientation import Orientation
import pytest

from kikuchipy.generators import EBSDSimulationGenerator


class TestEBSDSimulationGenerator:
    @pytest.mark.parametrize("nav_shape", [(5, 5), (25,), (1, 25)])
    def test_align_navigation_shape(
        self,
        nickel_phase,
        nordif_detector,
        nickel_rotations,
        r_tsl2bruker,
        nav_shape,
    ):
        """Check that initialization of a detector with orientations of
        a certain shape also reshapes the varying PCs, i.e. the detector
        navigation shape, if the detector has more than one PC.
        """
        assert nordif_detector.navigation_shape == (1,)
        nordif_detector.pc = np.ones((25, 3)) * nordif_detector.pc
        assert nordif_detector.navigation_shape == (25,)

        o_nickel = Orientation(r_tsl2bruker).set_symmetry(
            nickel_phase.point_group
        ) * nickel_rotations.reshape(*nav_shape)
        assert o_nickel.shape == nav_shape

        sim_gen = EBSDSimulationGenerator(
            phase=nickel_phase, detector=nordif_detector, orientations=o_nickel,
        )
        assert sim_gen.detector.navigation_shape == sim_gen.orientations.shape

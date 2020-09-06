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

from kikuchipy.detectors import EBSDDetector
from kikuchipy.generators import EBSDSimulationGenerator


class TestGeometricalEBSDSimulation:
    @pytest.mark.parametrize(
        "nordif_detector, nav_shape",
        [((5, 5), (5, 5))],
        indirect=["nordif_detector"],
    )
    def test_get_geometrical_ebsd_simulation_from_generator(
        self,
        nickel_phase,
        nordif_detector,
        nickel_rotations,
        r_tsl2bruker,
        nav_shape,
        nickel_rlp,
    ):
        """From generator works as expected overall."""
        assert isinstance(nordif_detector, EBSDDetector)
        o = Orientation(r_tsl2bruker).set_symmetry(
            nickel_phase.point_group
        ) * nickel_rotations.reshape(*nav_shape)
        sim_gen = EBSDSimulationGenerator(
            phase=nickel_phase, detector=nordif_detector, orientations=o
        )
        assert sim_gen.navigation_shape == (5, 5)

        sim = sim_gen.geometrical_simulation(nickel_rlp)
        assert np.allclose(sim.detector.pc, nordif_detector.pc)
        assert sim.detector.shape == nordif_detector.shape

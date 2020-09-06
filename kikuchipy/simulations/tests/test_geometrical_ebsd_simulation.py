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
from kikuchipy.generators import EBSDSimulationGenerator


class TestGeometricalEBSDSimulation:
    @pytest.mark.parametrize("nav_shape", [(5, 5), (25,), (1, 25), (25, 1)])
    def test_ebsd_simulation_navigation_shape(
        self, nickel_ebsd_simulation_generator, nickel_rlp, nav_shape,
    ):
        """Setting the navigation shape changes all derived navigation
        shapes.
        """
        sim_gen = nickel_ebsd_simulation_generator
        sim_gen.navigation_shape = nav_shape
        sim = sim_gen.geometrical_simulation(nickel_rlp)

        assert sim.detector.navigation_shape == nav_shape
        assert sim.rotations.shape == nav_shape
        assert sim.bands.navigation_shape == nav_shape

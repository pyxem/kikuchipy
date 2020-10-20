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
        simgen = nickel_ebsd_simulation_generator
        simgen.navigation_shape = nav_shape
        sim = simgen.geometrical_simulation(nickel_rlp)

        assert sim.detector.navigation_shape == nav_shape
        assert sim.rotations.shape == nav_shape
        assert sim.bands.navigation_shape == nav_shape
        assert sim.bands_detector_coordinates.shape == (
            nav_shape + (sim.bands.size, 4)
        )
        n_za = sim.zone_axes.size
        za_shape = nav_shape + (n_za, 2)
        assert sim.zone_axes_detector_coordinates.shape == za_shape
        assert sim.zone_axes_label_detector_coordinates.shape == za_shape
        assert sim.zone_axes_within_gnomonic_bounds.shape == nav_shape + (n_za,)

    def test_bands_detector_coordinates(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Desired band detector coordinates."""
        pass

    def test_zone_axes_coordinates(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Desired zone axes coordinates."""
        pass

    def test_zone_axes_label_detector_coordinates(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Desired zone axes label coordinates."""
        pass

    def test_bands_as_markers(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Desired line markers."""
        pass

    def test_zone_axes_as_markers(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Desired point markers."""
        pass

    def test_zone_axes_labels_as_markers(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Desired text markers."""
        pass

    def test_plot_zone_axes_labels_warns(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Matplotlib warns when plotting text with NaN coordinates."""
        pass

    def test_pc_as_markers(self, nickel_ebsd_simulation_generator, nickel_rlp):
        """Desired point markers."""
        pass

    def test_as_markers(self, nickel_ebsd_simulation_generator, nickel_rlp):
        """Desired set of markers."""
        pass

    def test_zone_axes_within_gnomonic_bounds(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Desired boolean array."""
        pass

    def test_repr(self, nickel_ebsd_simulation_generator, nickel_rlp):
        """Desired string representation."""
        pass

    def test_get_hkl_family(self, nickel_ebsd_simulation_generator, nickel_rlp):
        """Desired sets of families and indices."""
        pass

    def test_is_equivalent(self, nickel_ebsd_simulation_generator, nickel_rlp):
        """Desired equivalency, with reducing HKL."""
        pass

    def test_get_colors_for_allowed_bands(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Desired colors for bands."""
        pass

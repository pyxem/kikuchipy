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

import pytest

from kikuchipy.generators import EBSDSimulationGenerator


class TestEBSDSimulationGenerator:
    @pytest.mark.parametrize("nav_shape", [(5, 5), (25,), (1, 25)])
    def test_align_navigation_shape(
        self, nickel_phase, detector, nickel_rotations, r_tsl2bruker, nav_shape,
    ):
        """Initialization of a detector with orientations of a certain
        shape also reshapes the varying PCs, i.e. the detector
        navigation shape, if the detector has more than one PC.
        """
        assert detector.navigation_shape == (1,)
        o_nickel = r_tsl2bruker * nickel_rotations.reshape(*nav_shape)
        assert o_nickel.shape == nav_shape
        simgen = EBSDSimulationGenerator(
            detector=detector, phase=nickel_phase, rotations=o_nickel,
        )
        assert simgen.detector.navigation_shape == simgen.rotations.shape

    @pytest.mark.parametrize("nav_shape", [(5, 5), (25,), (1, 25), (25, 1)])
    def test_ebsd_simulation_generator_navigation_shape(
        self, nickel_ebsd_simulation_generator, nav_shape,
    ):
        """Setting the navigation shape changes all derived navigation
        shapes.
        """
        simgen = nickel_ebsd_simulation_generator
        assert simgen.navigation_shape == (25,)
        assert simgen.navigation_dimension == 1

        simgen.navigation_shape = nav_shape
        # sim_gen.navigation_shape is derived from sim.orientations.shape
        assert simgen.navigation_shape == nav_shape
        assert simgen.detector.navigation_shape == nav_shape

    @pytest.mark.parametrize(
        "shape_in, shape_change, ndim_in, ndim_change",
        [
            ((5, 5), (25,), 2, 1),
            ((25, 1), (1, 25), 2, 2),
            ((25,), (5, 5), 1, 2),
        ],
    )
    def test_set_rotations(
        self,
        nickel_phase,
        detector,
        nickel_rotations,
        shape_in,
        shape_change,
        ndim_in,
        ndim_change,
    ):
        """Setting rotations updates detector PC navigation shape."""
        r_nickel = nickel_rotations.reshape(*shape_in)
        simgen = EBSDSimulationGenerator(
            detector=detector, phase=nickel_phase, rotations=r_nickel
        )
        assert simgen.navigation_shape == shape_in
        assert simgen.detector.navigation_shape == shape_in
        assert simgen.navigation_dimension == ndim_in

        simgen.navigation_shape = shape_change
        assert simgen.navigation_shape == shape_change
        assert simgen.detector.navigation_shape == shape_change
        assert simgen.navigation_dimension == ndim_change

    def test_repr(self, nickel_ebsd_simulation_generator):
        """Desired string representation."""
        desired_repr = (
            "EBSDSimulationGenerator (25,)\n"
            "EBSDDetector (60, 60), px_size 70 um, binning 8, tilt 0, "
            "pc (0.421, 0.221, 0.505)\n"
            "<name: ni. space group: Fm-3m. point group: m-3m. "
            "proper point group: 432. color: tab:blue>\n"
            "Rotation (25,)\n"
        )
        assert repr(nickel_ebsd_simulation_generator) == desired_repr

    def test_geometrical_simulation(self, nickel_ebsd_simulation_generator):
        """Desired output EBSDGeometricalSimulation object."""
        pass

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

from diffpy.structure import Lattice
import numpy as np
from orix.quaternion import Rotation
import pytest

from kikuchipy.generators import EBSDSimulationGenerator
from kikuchipy.generators.ebsd_simulation_generator import (
    _get_coordinates_in_upper_hemisphere,
)


class TestEBSDSimulationGenerator:
    @pytest.mark.parametrize("nav_shape", [(5, 5), (25,), (1, 25)])
    def test_align_navigation_shape(
        self, nickel_phase, detector, nickel_rotations, nav_shape,
    ):
        """Initialization of a detector with orientations of a certain
        shape also reshapes the varying PCs, i.e. the detector
        navigation shape, if the detector has more than one PC.
        """
        assert detector.navigation_shape == (1,)
        o_nickel = nickel_rotations.reshape(*nav_shape)
        assert o_nickel.shape == nav_shape
        simgen = EBSDSimulationGenerator(
            detector=detector, phase=nickel_phase, rotations=o_nickel,
        )
        assert simgen.detector.navigation_shape == simgen.rotations.shape

    @pytest.mark.parametrize("nav_shape", [(5, 5), (25,), (1, 25), (25, 1)])
    def test_set_navigation_shape(
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

    @pytest.mark.parametrize("nav_shape", [(1, 2, 3), (1, 5, 6, 7)])
    def test_set_navigation_shape_raises(
        self, nickel_ebsd_simulation_generator, nav_shape
    ):
        """Cannot have a navigation dimension greater than 2."""
        with pytest.raises(ValueError, match="A maximum dimension of 2 is"):
            nickel_ebsd_simulation_generator.navigation_shape = nav_shape

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

    @pytest.mark.parametrize("nav_shape", [(5, 5, 1), (1, 5, 5, 1)])
    def test_set_rotations_raises(
        self, nickel_ebsd_simulation_generator, nav_shape,
    ):
        """Cannot have a navigation dimension greater than 2."""
        r = nickel_ebsd_simulation_generator.rotations.reshape(*nav_shape)
        with pytest.raises(ValueError, match="A maximum dimension of 2 is"):
            nickel_ebsd_simulation_generator.rotations = r

    def test_align_pc_with_rotations_shape_raises(
        self, nickel_ebsd_simulation_generator,
    ):
        """Detector and rotations navigation shapes must be compatible."""
        simgen = nickel_ebsd_simulation_generator
        simgen.detector.pc = simgen.detector.pc[:2]
        with pytest.raises(ValueError, match="The detector navigation shape"):
            simgen._align_pc_with_rotations_shape()

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

    @pytest.mark.parametrize(
        "nav_idx, desired_shape, desired_pc, desired_rotation",
        [
            ((2, 3), (1,), [0.1, 0.2, 0.3], [1, 2, 3]),
            ((slice(0, 2), 0), (2,), [0.5, 0.2, 0.3], [1, 2, 3]),
            ((slice(0, 2), slice(2, 4)), (2, 2), [0.1, 0.7, 0.3], [1, 2, 3]),
        ],
    )
    def test_get_item(
        self,
        nickel_ebsd_simulation_generator,
        nav_idx,
        desired_shape,
        desired_pc,
        desired_rotation,
    ):
        """Desired PC and rotation values from __getitem__()."""
        simgen = nickel_ebsd_simulation_generator
        simgen.navigation_shape = (5, 5)
        simgen.detector.pc[nav_idx] = desired_pc
        r = Rotation.from_euler(desired_rotation)
        simgen.rotations[nav_idx] = r

        simgen2 = simgen[nav_idx]
        assert simgen2.navigation_shape == desired_shape
        assert np.allclose(simgen2.detector.pc, desired_pc)
        assert np.allclose(simgen2.rotations.data, r.data)

        new_pc = [0.5,] * 3
        simgen2.detector.pc[0] = new_pc
        assert not np.allclose(simgen[nav_idx].detector.pc, simgen2.detector.pc)

    def test_geometrical_simulation2d(
        self, nickel_ebsd_simulation_generator, nickel_rlp,
    ):
        """Desired output EBSDGeometricalSimulation object."""
        simgen = nickel_ebsd_simulation_generator

        rlp = nickel_rlp[nickel_rlp.allowed].symmetrise()
        assert rlp.size == 26

        sim1 = simgen.geometrical_simulation(rlp)
        assert np.allclose(sim1.detector.pc, simgen.detector.pc)
        assert np.allclose(
            sim1.bands.phase.point_group.data, simgen.phase.point_group.data
        )
        assert np.allclose(sim1.rotations.data, simgen.rotations.data)
        assert sim1.bands.navigation_shape == simgen.navigation_shape
        assert sim1.bands.size == 18

        sim2 = simgen.geometrical_simulation()
        assert sim2.bands.size == 132

    def test_geometrical_simulation1d(
        self, nickel_ebsd_simulation_generator, nickel_rlp,
    ):
        """Geometrical EBSD simulations handle 1d."""
        simgen = nickel_ebsd_simulation_generator[:5]

        rlp = nickel_rlp[nickel_rlp.allowed].symmetrise()
        assert rlp.size == 26

        sim1 = simgen.geometrical_simulation(rlp)
        assert np.allclose(sim1.detector.pc, simgen.detector.pc)
        assert np.allclose(
            sim1.bands.phase.point_group.data, simgen.phase.point_group.data
        )
        assert np.allclose(sim1.rotations.data, simgen.rotations.data)
        assert sim1.bands.navigation_shape == (5,)
        assert sim1.bands.size == 15

    def test_geometrical_simulation0d(
        self, nickel_ebsd_simulation_generator, nickel_rlp,
    ):
        """Geometrical EBSD simulations handle 0d."""
        simgen = nickel_ebsd_simulation_generator[0]
        simgen.navigation_shape = ()

        rlp = nickel_rlp[nickel_rlp.allowed].symmetrise()
        assert rlp.size == 26

        sim1 = simgen.geometrical_simulation(rlp)
        assert np.allclose(sim1.detector.pc, simgen.detector.pc)
        assert np.allclose(
            sim1.bands.phase.point_group.data, simgen.phase.point_group.data
        )
        assert np.allclose(sim1.rotations.data, simgen.rotations.data)
        assert sim1.bands.navigation_shape == ()
        assert sim1.bands.size == 13

    def test_geometrical_simulation_raises(
        self, nickel_ebsd_simulation_generator,
    ):
        """Generator must have a Phase object with point group
        symmetries and a crystal structure if no ReciprocalLatticePoint
        object is passed when simulating geometrical EBSD patterns.
        """
        simgen = nickel_ebsd_simulation_generator
        simgen.phase.space_group = None
        with pytest.raises(ValueError, match="A ReciprocalLatticePoint object"):
            _ = simgen.geometrical_simulation()

    @pytest.mark.parametrize(
        "space_group, lattice, compatible",
        [
            (225, Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90), True),
            (200, Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90), False),
            (225, Lattice(3.5246, 3.5236, 3.5236, 90, 90, 90), False),
        ],
    )
    def test_rlp_phase_is_compatible(
        self,
        nickel_ebsd_simulation_generator,
        nickel_rlp,
        space_group,
        lattice,
        compatible,
    ):
        """Desired behaviour when checking whether the simulation
        generator phase and the ReciprocalLatticePoint phase are
        compatible.
        """
        simgen = nickel_ebsd_simulation_generator
        nickel_rlp.phase.space_group = space_group
        nickel_rlp.phase.structure.lattice = lattice

        if compatible:
            assert simgen._rlp_phase_is_compatible(nickel_rlp) is None
        else:
            with pytest.raises(ValueError, match="The lattice parameters and/"):
                simgen._rlp_phase_is_compatible(nickel_rlp)

    def test_band_coordinates(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Desired band values like detector coordinates."""
        simgen = nickel_ebsd_simulation_generator
        assert simgen.navigation_shape == (25,)

        rlp = nickel_rlp[nickel_rlp.allowed].symmetrise()
        sim = simgen[:2].geometrical_simulation(rlp)

        bands = sim.bands
        assert bands.size == 14
        assert np.allclose(bands.hkl.data[5:7], [[0, -2, 0], [0, 0, 2]])
        assert np.allclose(
            bands.hkl_detector[:, 5:7].data,
            np.array(
                [
                    [
                        [-0.42182878, -0.31973595, 0.20494058],
                        [0.31657776, -0.12684294, 0.45371866],
                    ],
                    [
                        [-0.42871728, -0.14366351, 0.3431232],
                        [0.37003555, -0.1111772, 0.41579389],
                    ],
                ]
            ),
        )
        assert np.allclose(bands.gnomonic_radius, simgen[:2].detector.r_max)

    def test_get_coordinates_in_upper_hemisphere_2d(self):
        """Coordinates are considered correctly whether to be in the
        upper hemisphere and visible in a point (pattern).
        """
        # Shape (2, 2, 2, 3): (n bands/zone axes, i, j, xyz)
        coords = np.array(
            [
                [[[1, 2, -1], [1, 2, -2]], [[1, 2, -3], [1, 2, 0]]],
                [[[1, 2, -3], [1, 2, 3]], [[1, 2, 1e-3], [1, 2, -3]]],
            ]
        )
        n_nav_dims = 2
        navigation_axes = (1, 2)[:n_nav_dims]
        is_upper, in_a_point = _get_coordinates_in_upper_hemisphere(
            z_coordinates=coords[..., 2], navigation_axes=navigation_axes
        )
        assert np.allclose(
            is_upper,
            [[[False, False], [False, False]], [[False, True], [True, False]]],
        )
        assert np.allclose(in_a_point, [False, True])

    def test_get_coordinates_in_upper_hemisphere_1d(self):
        """Coordinates are considered correctly whether to be in the
        upper hemisphere and visible in a point (pattern).
        """
        # Shape (2, 2, 3): (n bands/zone axes, i, xyz)
        coords = np.array([[[1, 2, -1], [1, 2, -2]], [[1, 2, -3], [1, 2, 3]]])
        n_nav_dims = 1
        navigation_axes = (1, 2)[:n_nav_dims]
        is_upper, in_a_point = _get_coordinates_in_upper_hemisphere(
            z_coordinates=coords[..., 2], navigation_axes=navigation_axes
        )

        assert np.allclose(is_upper, [[False, False], [False, True]])
        assert np.allclose(in_a_point, [False, True])

    def test_get_coordinates_in_upper_hemisphere_0d(self):
        """Coordinates are considered correctly whether to be in the
        upper hemisphere and visible in a point (pattern).
        """
        # Shape (2, 3): (n bands/zone axes, xyz)
        coords = np.array([[1, 2, -1], [1, 2, 2]])
        n_nav_dims = 0
        navigation_axes = (1, 2)[:n_nav_dims]
        is_upper, in_a_point = _get_coordinates_in_upper_hemisphere(
            z_coordinates=coords[..., 2], navigation_axes=navigation_axes
        )

        assert np.allclose(is_upper, [False, True])
        assert np.allclose(in_a_point, [False, True])

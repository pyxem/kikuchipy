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

from diffpy.structure import Atom, Lattice, Structure
from diffsims.crystallography import ReciprocalLatticeVector
import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import Phase
from orix.plot import StereographicPlot
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp


def _setup_method():
    phase = Phase(
        space_group=225,
        structure=Structure(
            atoms=[Atom("Al", [0, 0, 0])],
            lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90),
        ),
    )
    ref = ReciprocalLatticeVector(
        phase, hkl=((1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1))
    )
    ref.sanitise_phase()
    ref = ref.symmetrise()
    ref.calculate_structure_factor()
    ref.calculate_theta(20e3)
    return kp.simulations.KikuchiPatternSimulator(ref)


class TestKikuchiPatternSimulator:
    def test_init_attributes_repr(self, nickel_phase):
        ref = ReciprocalLatticeVector(
            nickel_phase, hkl=((1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1))
        ).reshape(2, 2)
        simulator1 = kp.simulations.KikuchiPatternSimulator(ref)

        assert simulator1.reflectors.shape == (4,)
        assert not np.may_share_memory(ref.hkl.data, simulator1.reflectors.hkl.data)
        with pytest.raises(ValueError, match="Reflectors have no Bragg angles."):
            simulator1._raise_if_no_theta()
        with pytest.raises(ValueError, match="Reflectors have no structure factors."):
            simulator1._raise_if_no_structure_factor()
        assert repr(simulator1) == (
            "KikuchiPatternSimulator (4,), ni (m-3m)\n"
            "[[1. 1. 1.]\n"
            " [2. 2. 0.]\n"
            " [2. 0. 0.]\n"
            " [3. 1. 1.]]"
        )

        ref = ref.flatten()
        ref.sanitise_phase()
        ref = ref.symmetrise()
        ref.calculate_structure_factor()
        ref.calculate_theta(20e3)
        simulator2 = kp.simulations.KikuchiPatternSimulator(ref)

        assert simulator2.reflectors.shape == (50,)
        assert simulator2.reflectors.phase.point_group == ref.phase.point_group
        # Does not raise errors
        simulator2._raise_if_no_theta()
        simulator2._raise_if_no_structure_factor()


class TestCalculateMasterPattern:
    def setup_method(self):
        self.simulator = _setup_method()

    def test_default(self):
        simulator = self.simulator
        mp = simulator.calculate_master_pattern()

        assert isinstance(mp, kp.signals.EBSDMasterPattern)
        assert mp.data.shape == (1001, 1001)
        assert mp.mode == "kinematical"

    def test_raises(self):
        simulator = self.simulator
        with pytest.raises(ValueError, match="Unknown `hemisphere`, valid options are"):
            _ = simulator.calculate_master_pattern(hemisphere="north")
        with pytest.raises(ValueError, match="Unknown `scaling`, valid options are"):
            _ = simulator.calculate_master_pattern(scaling="cubic")

    def test_shape(self):
        simulator = self.simulator
        mp = simulator.calculate_master_pattern(half_size=100, hemisphere="both")
        assert mp.data.shape == (2, 201, 201)
        assert np.allclose(mp.data[0], mp.data[1])

    def test_scaling(self):
        simulator = self.simulator

        mp1 = simulator.calculate_master_pattern(half_size=100, scaling="linear")
        assert np.isclose(mp1.data.mean(), 3.53, atol=1e-2)

        mp2 = simulator.calculate_master_pattern(half_size=100, scaling="square")
        assert np.isclose(mp2.data.mean(), 19.00, atol=1e-2)

        mp3 = simulator.calculate_master_pattern(half_size=100, scaling=None)
        assert np.isclose(mp3.data.mean(), 0.74, atol=1e-2)


class TestOnDetector:
    def setup_method(self):
        self.simulator = _setup_method()
        self.detector = kp.detectors.EBSDDetector(shape=(60, 60))

    def test_1d(self):
        rot1 = Rotation.random()
        sim1 = self.simulator.on_detector(self.detector, rot1)
        assert np.allclose(sim1.rotations.data, rot1.data)
        assert np.allclose(sim1.detector.pc, self.detector.pc)
        assert sim1.navigation_shape == rot1.shape == (1,)
        sim1.plot()

        rot2 = Rotation.random((10,))
        sim2 = self.simulator.on_detector(self.detector, rot2)
        assert np.allclose(sim2.rotations.data, rot2.data)
        assert np.allclose(sim2.detector.pc, self.detector.pc)
        assert sim2.navigation_shape == rot2.shape == (10,)
        sim2.plot(5)

        plt.close("all")

    def test_2d(self):
        rot1 = Rotation.random((3, 1))
        sim1 = self.simulator.on_detector(self.detector, rot1)
        assert np.allclose(sim1.rotations.data, rot1.data)
        assert np.allclose(sim1.detector.pc, self.detector.pc)
        assert sim1.navigation_shape == rot1.shape == (3, 1)
        sim1.plot((0, 0))

        rot2 = Rotation.random((2, 2))
        sim2 = self.simulator.on_detector(self.detector, rot2)
        assert np.allclose(sim2.rotations.data, rot2.data)
        assert np.allclose(sim2.detector.pc, self.detector.pc)
        assert sim2.navigation_shape == rot2.shape == (2, 2)
        sim2.plot((1, 1))

        plt.close("all")


class TestPlot:
    def setup_method(self):
        self.simulator = _setup_method()

    def test_default(self):
        simulator = self.simulator
        ref = simulator.reflectors.unique(use_symmetry=True)
        simulator._reflectors = ref
        assert simulator.reflectors.size == 4

        fig = simulator.plot(return_figure=True)
        ax = fig.axes[0]
        assert isinstance(ax, StereographicPlot)
        assert ax.pole == -1
        assert len(ax.lines) == 4

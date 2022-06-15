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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import Phase
from orix.plot import StereographicPlot
from orix.quaternion import Rotation
from packaging.version import Version
import pytest

import kikuchipy as kp


def _setup_method():
    """Return simulator used in `setup_method` of multiple test classes."""
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
    """General features of the KikuchiPatternSimulator class."""

    def test_init_attributes_repr(self, nickel_phase):
        """Initialization, attributes and string representation."""
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
            "KikuchiPatternSimulator:\n"
            "ReciprocalLatticeVector (4,), ni (m-3m)\n"
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
    """Calculation of master pattern."""

    def setup_method(self):
        self.simulator = _setup_method()

    def test_default(self):
        """Default values are as expected."""
        simulator = self.simulator
        mp = simulator.calculate_master_pattern()

        assert isinstance(mp, kp.signals.EBSDMasterPattern)
        assert mp.data.shape == (1001, 1001)

    def test_raises(self):
        """Appropriate error messages are raised."""
        simulator = self.simulator
        with pytest.raises(ValueError, match="Unknown `hemisphere`, options are"):
            _ = simulator.calculate_master_pattern(hemisphere="north")
        with pytest.raises(ValueError, match="Unknown `scaling`, options are"):
            _ = simulator.calculate_master_pattern(scaling="cubic")

    def test_shape(self):
        """Output shape as expected."""
        simulator = self.simulator
        mp = simulator.calculate_master_pattern(half_size=100, hemisphere="both")
        assert mp.data.shape == (2, 201, 201)
        assert np.allclose(mp.data[0], mp.data[1])

        axes_names = [a["name"] for a in mp.axes_manager.as_dictionary().values()]
        assert axes_names == ["hemisphere", "height", "width"]

    def test_scaling(self):
        """Scaling options give expected output intensities."""
        simulator = self.simulator

        mp1 = simulator.calculate_master_pattern(half_size=100)
        assert np.isclose(mp1.data.mean(), 3.53, atol=1e-2)
        mp2 = simulator.calculate_master_pattern(half_size=100, hemisphere="lower")
        assert np.isclose(mp2.data.mean(), 3.53, atol=1e-2)

        mp3 = simulator.calculate_master_pattern(half_size=100, scaling="square")
        assert np.isclose(mp3.data.mean(), 19.00, atol=1e-2)

        mp4 = simulator.calculate_master_pattern(half_size=100, scaling=None)
        assert np.isclose(mp4.data.mean(), 0.74, atol=1e-2)


class TestOnDetector:
    """Test determination of detector coordinates of geometrical
    simulations given a detector and rotation(s).

    """

    def setup_method(self):
        self.simulator = _setup_method()
        self.detector = kp.detectors.EBSDDetector(shape=(60, 60))

    def test_1d(self):
        """1D rotation instance works."""
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
        """2D rotation instance works."""
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

    def test_raises_incompatible_shapes(self):
        detector = self.detector
        detector.pc = np.full((2, 3), detector.pc)
        with pytest.raises(ValueError, match="`detector.navigation_shape` is not "):
            _ = self.simulator.on_detector(detector, Rotation.random((3, 2)))


class TestPlot:
    """Test plot method."""

    def setup_method(self):
        self.simulator = _setup_method()

    def test_default(self):
        """Default values are as expected, and appropriate errors are
        raised.
        """
        simulator = self.simulator
        ref = simulator.reflectors.unique(use_symmetry=True)
        simulator._reflectors = ref
        assert simulator.reflectors.size == 4

        fig = simulator.plot(return_figure=True)
        ax = fig.axes[0]
        assert isinstance(ax, StereographicPlot)
        assert ax.pole == -1
        assert len(ax.lines) == 4

        with pytest.raises(ValueError, match="Unknown `projection`, options are "):
            simulator.plot("lambert")

        plt.close("all")

    def test_modes(self):
        """Modes 'lines' and 'bands' works as expected, and appropriate
        error is raised.
        """
        simulator = self.simulator
        ref = simulator.reflectors.unique(use_symmetry=True)
        simulator._reflectors = ref  # Hack
        assert simulator.reflectors.size == 4

        # Correct number of lines added
        fig1 = simulator.plot(mode="lines", return_figure=True)
        assert len(fig1.axes[0].lines) == 4
        fig2 = simulator.plot(mode="bands", return_figure=True)
        assert len(fig2.axes[0].lines) == 8

        # Adding to existing figure
        simulator.plot(mode="lines", figure=fig2)
        assert len(fig2.axes[0].lines) == 12
        simulator.plot(mode="bands", figure=fig1)
        assert len(fig1.axes[0].lines) == 12

        plt.close("all")

        # Raises errors
        simulator._reflectors._theta = np.full(ref.size, np.nan)  # Hack
        with pytest.raises(ValueError, match="Requires that reflectors have Bragg"):
            _ = simulator.plot(mode="bands")
        with pytest.raises(ValueError, match="Unknown `mode`, options are"):
            _ = simulator.plot(mode="kinematical")

    def test_hemisphere(self):
        """Plotting upper, lower, or both hemispheres works, also
        extending either of these by passing the figure on.
        """
        simulator = self.simulator
        fig1 = simulator.plot(hemisphere="upper", return_figure=True)
        assert len(fig1.axes) == 1
        assert fig1.axes[0].pole == -1

        fig2 = simulator.plot(hemisphere="lower", return_figure=True)
        assert len(fig2.axes) == 1
        assert fig2.axes[0].pole == 1

        fig3 = simulator.plot(hemisphere="both", return_figure=True)
        assert len(fig3.axes) == 2
        assert fig3.axes[0].pole == -1
        # Four [200] vectors and four [220] vectors lie on the equator,
        # so 29 instead of just half of the vectors, 50, are included in
        # each hemisphere
        assert len(fig3.axes[0].lines) == 29
        assert len(fig3.axes[0].lines) == 29
        # Passing a figure with more than one hemisphere adds all
        # vectors to all hemispheres
        simulator.plot(figure=fig3)
        assert len(fig3.axes[0].lines) == 29 * 2
        assert len(fig3.axes[0].lines) == 29 * 2

        plt.close("all")

    def test_spherical(self):
        """Spherical plot with Matplotlib."""
        simulator = self.simulator
        fig1 = simulator.plot("spherical", return_figure=True)
        ax1 = fig1.axes[0]
        assert ax1.name == "3d"
        assert len(ax1.lines) == simulator.reflectors.size

        fig2 = simulator.plot("spherical", mode="bands", return_figure=True)
        ax2 = fig2.axes[0]
        assert len(ax2.lines) == simulator.reflectors.size * 2

        simulator.plot("spherical", figure=fig2)
        assert len(ax2.lines) == simulator.reflectors.size * 3
        if Version(matplotlib.__version__) < Version("3.5.0"):
            assert len(ax2.artists) == 6  # pragma: no cover
        else:
            assert len(ax2.patches) == 6  # Reference frame arrows added twice...

        plt.close("all")

    @pytest.mark.skipif(not kp._pyvista_installed, reason="Pyvista is not installed")
    def test_spherical_pyvista(self):
        """Spherical plot with PyVista."""
        import pyvista as pv

        simulator = self.simulator
        fig1 = simulator.plot(
            "spherical", backend="pyvista", return_figure=True, show_plotter=False
        )
        assert isinstance(fig1, pv.Plotter)
        assert isinstance(fig1.mesh, pv.PolyData)
        assert fig1.mesh.n_cells == simulator.reflectors.size
        assert np.allclose(fig1.mesh.bounds, [-1, 1, -1, 1, -1, 1])

        fig2 = simulator.plot("spherical", backend="pyvista", return_figure=True)
        with pytest.raises(RuntimeError, match="This plotter has been closed "):
            fig2.show()

        # Add to existing Plotter
        simulator.plot(
            "spherical",
            backend="pyvista",
            mode="bands",
            show_plotter=False,
            figure=fig1,
        )
        assert fig1.mesh.n_cells == simulator.reflectors.size * 2

        plt.close("all")

    @pytest.mark.skipif(kp._pyvista_installed, reason="Pyvista is installed")
    def test_spherical_pyvista_raises(self):  # pragma: no cover
        """Appropriate error message is raised when PyVista is
        unavailable.
        """
        with pytest.raises(ImportError, match="Pyvista is not installed"):
            _ = self.simulator.plot("spherical", backend="pyvista")

    def test_scaling(self):
        """Intensity scaling works as expected."""
        simulator = self.simulator

        # Linear
        fig1 = simulator.plot(return_figure=True)
        colors1 = np.stack([line.get_color() for line in fig1.axes[0].lines])
        unique_colors1 = np.unique(colors1.round(6), axis=0)
        assert unique_colors1.shape[0] == 4

        # Square
        fig2 = simulator.plot(scaling="square", return_figure=True)
        colors2 = np.stack([line.get_color() for line in fig2.axes[0].lines])
        unique_colors2 = np.unique(colors2.round(6), axis=0)
        assert unique_colors1.mean() > unique_colors2.mean()

        # None
        fig3 = simulator.plot(scaling=None, return_figure=True)
        colors3 = np.stack([line.get_color() for line in fig3.axes[0].lines])
        unique_colors3 = np.unique(colors3.round(6), axis=0)
        assert unique_colors3.shape[0] == 1

        with pytest.raises(ValueError, match="Unknown `scaling`, options are "):
            _ = simulator.plot(scaling="cubic")

#
# Copyright 2019-2026 the kikuchipy developers
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
#

import pytest

ipywidgets = pytest.importorskip("ipywidgets")

from diffpy.structure import Atom, Lattice, Structure
from diffsims.crystallography import ReciprocalLatticeVector
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np
import orix.crystal_map as ocm
import orix.quaternion as oqu

import kikuchipy as kp
from kikuchipy.draw._ebsd_detector_plot_widgets import (
    EBSDDetectorPlotter,
    GeometricalSimulationOverlay,
    MasterPatternOverlay,
    combine_widgets,
    get_detector_azimuthal_slider,
    get_detector_twist_slider,
    get_detector_value_range,
    get_pcx_slider,
    get_pcy_slider,
    get_pcz_slider,
    get_phi1_slider,
    get_phi2_slider,
    get_Phi_slider,
    plot_detector_sample_geometry_side_view_interactive,
    plot_detector_sample_geometry_top_view_interactive,
)


class TestEBSDDetectorPlotWidgets:
    def test_get_detector_value_range(self):

        # Value within range: unchanged
        vmin, vmax = get_detector_value_range(5.0, 0.0, 10.0)
        assert vmin == 0.0
        assert vmax == 10.0

        # Value below range: vmin is expanded
        vmin, vmax = get_detector_value_range(-5.0, 0.0, 10.0)
        assert vmin < -5.0
        assert vmax == 10.0

        # Value above range: vmax is expanded
        vmin, vmax = get_detector_value_range(15.0, 0.0, 10.0)
        assert vmin == 0.0
        assert vmax > 15.0

    def test_get_detector_azimuthal_slider(self):
        det = kp.detectors.EBSDDetector(azimuthal=5.0)
        slider = get_detector_azimuthal_slider(det)
        assert isinstance(slider, ipywidgets.FloatSlider)
        assert slider.value == 5.0

    def test_get_detector_twist_slider(self):
        det = kp.detectors.EBSDDetector(twist=3.0)
        slider = get_detector_twist_slider(det)
        assert isinstance(slider, ipywidgets.FloatSlider)
        assert slider.value == 3.0

    def test_get_pc_sliders(self):
        det = kp.detectors.EBSDDetector(pc=[0.4, 0.5, 0.6])

        pcx_slider = get_pcx_slider(det)
        assert isinstance(pcx_slider, ipywidgets.FloatSlider)
        assert np.isclose(pcx_slider.value, 0.4)

        pcy_slider = get_pcy_slider(det)
        assert isinstance(pcy_slider, ipywidgets.FloatSlider)
        assert np.isclose(pcy_slider.value, 0.5)

        pcz_slider = get_pcz_slider(det)
        assert isinstance(pcz_slider, ipywidgets.FloatSlider)
        assert np.isclose(pcz_slider.value, 0.6)

    def test_get_rotation_sliders(self):
        phi1 = get_phi1_slider(10.0)
        Phi = get_Phi_slider(20.0)
        phi2 = get_phi2_slider(30.0)

        assert isinstance(phi1, ipywidgets.FloatSlider)
        assert isinstance(Phi, ipywidgets.FloatSlider)
        assert isinstance(phi2, ipywidgets.FloatSlider)
        assert phi1.value == 10.0
        assert Phi.value == 20.0
        assert phi2.value == 30.0

    def test_combine_widgets(self):
        det_widgets = [ipywidgets.FloatSlider()]
        rot_widgets = [ipywidgets.FloatSlider()]

        # Without simulation checkbox -> VBox (only detector column)
        controls = combine_widgets(det_widgets, rot_widgets, None)
        assert isinstance(controls, ipywidgets.VBox)

        # With simulation checkbox -> HBox (detector + rotation + sim columns)
        checkbox = ipywidgets.Checkbox()
        controls = combine_widgets(det_widgets, rot_widgets, checkbox)
        assert isinstance(controls, ipywidgets.HBox)

    def test_plot_detector_sample_geometry_side_view_interactive(self):
        det = kp.detectors.EBSDDetector()
        controls, fig = plot_detector_sample_geometry_side_view_interactive(det)

        assert isinstance(controls, ipywidgets.VBox)
        assert isinstance(fig, mfigure.Figure)
        plt.close("all")

    def test_plot_detector_sample_geometry_top_view_interactive(self):
        det = kp.detectors.EBSDDetector()
        controls, fig = plot_detector_sample_geometry_top_view_interactive(det)

        assert isinstance(controls, ipywidgets.VBox)
        assert isinstance(fig, mfigure.Figure)
        plt.close("all")


class TestEBSDDetectorPlotter:
    def test_create_ebsd_detector_plotter(self):
        det = kp.detectors.EBSDDetector()

        # Default: deep copy
        plotter = EBSDDetectorPlotter(det)
        assert plotter._detector is not det
        assert plotter._overlays == []

        # inplace=True: no copy
        plotter2 = EBSDDetectorPlotter(det, inplace=True)
        assert plotter2._detector is det

    def test_add_geometrical_simulation(self):
        phase = ocm.Phase(
            space_group=225,
            structure=Structure(
                atoms=[Atom("Ni", [0, 0, 0])],
                lattice=Lattice(3.524, 3.524, 3.524, 90, 90, 90),
            ),
        )
        ref = ReciprocalLatticeVector(phase, hkl=[[1, 1, 1], [2, 0, 0], [2, 2, 0]])
        rot = oqu.Rotation.from_axes_angles([0, 0, 1], 0)

        plotter = EBSDDetectorPlotter(kp.detectors.EBSDDetector(), rotation=rot)
        plotter.set_geometrical_simulation(ref)
        assert len(plotter._overlays) == 1
        assert isinstance(plotter._overlays[0], GeometricalSimulationOverlay)

        fig, controls = plotter.show()
        assert isinstance(fig, mfigure.Figure)
        assert isinstance(controls, ipywidgets.HBox)
        assert len(fig.axes) == 3
        plt.close("all")

        # Raises when plotter has no rotation
        plotter2 = EBSDDetectorPlotter(kp.detectors.EBSDDetector())
        with pytest.raises(RuntimeError, match="Plotter must be made with a rotation"):
            plotter2.set_geometrical_simulation(ref)

    def test_set_master_pattern(self):
        mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
        rot = oqu.Rotation.from_axes_angles([0, 0, 1], 0)

        plotter = EBSDDetectorPlotter(kp.detectors.EBSDDetector(), rotation=rot)
        plotter.set_master_pattern(mp)
        assert len(plotter._overlays) == 1
        assert isinstance(plotter._overlays[0], MasterPatternOverlay)

        fig, controls = plotter.show()
        assert isinstance(fig, mfigure.Figure)
        assert isinstance(controls, ipywidgets.HBox)
        assert len(fig.axes) == 3
        plt.close("all")

        # Raises if there's already a master pattern added
        with pytest.raises(ValueError, match="Plotter already has a master pattern"):
            plotter.set_master_pattern(mp)

        # Raises when plotter has no rotation
        plotter2 = EBSDDetectorPlotter(kp.detectors.EBSDDetector())
        with pytest.raises(RuntimeError, match="Plotter must be made with a rotation"):
            plotter2.set_master_pattern(mp)

    def test_show_ebsd_detector_plotter(self):
        det = kp.detectors.EBSDDetector(shape=(60, 60))
        plotter = EBSDDetectorPlotter(det)
        fig, controls = plotter.show()

        assert isinstance(fig, mfigure.Figure)
        assert isinstance(controls, ipywidgets.HBox)
        assert len(fig.axes) == 3

        plt.close("all")

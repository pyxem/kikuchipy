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

from hyperspy.utils.markers import line_segment, point, text
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kikuchipy.signals import EBSD


matplotlib.use("Agg")  # For plotting


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

    @pytest.mark.parametrize(
        "nav_slice, nav_shape, n_bands, n_za",
        [
            ((0, 0), (), 4, 6),  # 0d
            ((0, slice(0, 3)), (3,), 4, 7),  # 1d
            ((slice(0, 4), slice(0, 2)), (4, 2), 4, 7),  # 2d
        ],
    )
    def test_detector_coordinates_shapes(
        self,
        nickel_ebsd_simulation_generator,
        nickel_rlp,
        nav_slice,
        nav_shape,
        n_bands,
        n_za,
    ):
        """Desired detector coordinates without a navigation space."""
        simgen = nickel_ebsd_simulation_generator
        simgen.navigation_shape = (5, 5)
        rlp = nickel_rlp.symmetrise()

        simgen = simgen[nav_slice]
        if nav_shape == ():  # Have to reshape data, otherwise equals (1,)
            simgen.navigation_shape = nav_shape
        sim = simgen.geometrical_simulation(rlp[:10])
        assert sim.bands.navigation_shape == nav_shape
        assert sim.zone_axes.navigation_shape == nav_shape
        assert sim.bands._data_shape == nav_shape + (n_bands,)
        assert sim.zone_axes._data_shape == nav_shape + (n_za,)
        assert sim.bands_detector_coordinates.shape == (
            nav_shape + (n_bands, 4)
        )
        assert sim.zone_axes_detector_coordinates.shape == (
            nav_shape + (n_za, 2)
        )
        assert sim.zone_axes_label_detector_coordinates.shape == (
            nav_shape + (n_za, 2)
        )

    @pytest.mark.parametrize("exclude, not_nan", [(True, 376), (False, 678)])
    def test_zone_axes_detector_coordinates_exclude(
        self, nickel_ebsd_simulation_generator, nickel_rlp, exclude, not_nan
    ):
        """Desired number of zone axes if excluding outside detector."""
        simgen = nickel_ebsd_simulation_generator
        sim = simgen.geometrical_simulation(nickel_rlp.symmetrise())

        sim.exclude_outside_detector = exclude
        assert sim.zone_axes_detector_coordinates.shape == (
            simgen.navigation_shape + (35, 2)
        )
        assert (
            np.sum(np.isfinite(sim.zone_axes_detector_coordinates)) == not_nan
        )
        assert sim.exclude_outside_detector == exclude

    @pytest.mark.parametrize("nav_shape", [(5, 5), (1, 25), (25, 1), (25,)])
    def test_bands_as_markers(
        self, nickel_ebsd_simulation_generator, nickel_rlp, nav_shape
    ):
        """Line markers work."""
        simgen = nickel_ebsd_simulation_generator
        simgen.navigation_shape = nav_shape
        sim = simgen.geometrical_simulation(nickel_rlp.symmetrise())

        se = EBSD(np.ones(nav_shape + (60, 60), dtype=np.uint8))

        se.add_marker(
            marker=sim.bands_as_markers(), permanent=True, plot_marker=False
        )
        assert isinstance(se.metadata.Markers.line_segment, line_segment)
        se.plot()
        plt.close("all")

    def test_bands_as_markers_family_colors(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Line markers work when specifying colors."""
        simgen = nickel_ebsd_simulation_generator
        sim = simgen.geometrical_simulation(nickel_rlp[:2])

        se = EBSD(np.ones(simgen.navigation_shape + (60, 60), dtype=np.uint8))

        colors = ["lime", "tab:blue"]
        se.add_marker(
            marker=sim.bands_as_markers(family_colors=colors),
            permanent=True,
            plot_marker=False,
        )
        assert (
            se.metadata.Markers.line_segment.marker_properties["color"]
            == colors[0]
        )
        assert (
            se.metadata.Markers.line_segment1.marker_properties["color"]
            == colors[1]
        )
        se.plot()
        plt.close("all")

    def test_bands_as_markers_1d_nav(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """1D nav shape band markers work."""
        simgen = nickel_ebsd_simulation_generator[0]
        assert simgen.navigation_shape == (1,)
        sim = simgen.geometrical_simulation(nickel_rlp.symmetrise())
        assert sim.bands.navigation_shape == (1,)

        se = EBSD(np.ones((60, 60), dtype=np.uint8))

        se.add_marker(
            marker=sim.bands_as_markers(), permanent=False, plot_marker=True,
        )
        plt.close("all")

    @pytest.mark.parametrize("nav_shape", [(5, 5), (1, 25), (25, 1), (25,)])
    def test_zone_axes_as_markers(
        self, nickel_ebsd_simulation_generator, nickel_rlp, nav_shape
    ):
        """Point markers work."""
        simgen = nickel_ebsd_simulation_generator
        simgen.navigation_shape = nav_shape
        sim = simgen.geometrical_simulation(nickel_rlp.symmetrise())

        se = EBSD(np.ones(nav_shape + (60, 60), dtype=np.uint8))

        se.add_marker(
            marker=sim.zone_axes_as_markers(), permanent=True, plot_marker=False
        )
        assert isinstance(se.metadata.Markers.point, point)
        se.plot()
        plt.close("all")

    @pytest.mark.parametrize("nav_shape", [(5, 5), (1, 25), (25, 1), (25,)])
    def test_zone_axes_labels_as_markers(
        self, nickel_ebsd_simulation_generator, nickel_rlp, nav_shape
    ):
        """Text markers work."""
        simgen = nickel_ebsd_simulation_generator
        simgen.navigation_shape = nav_shape
        sim = simgen.geometrical_simulation(nickel_rlp.symmetrise())

        se = EBSD(np.ones(nav_shape + (60, 60), dtype=np.uint8))

        matplotlib.set_loglevel("error")
        se.add_marker(
            marker=sim.zone_axes_labels_as_markers(),
            permanent=True,
            plot_marker=False,
        )
        assert isinstance(se.metadata.Markers.text, text)
        se.plot()
        plt.close("all")
        matplotlib.set_loglevel("warning")

    def test_plot_zone_axes_labels_warns(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Matplotlib warns when plotting text with NaN coordinates."""
        simgen = nickel_ebsd_simulation_generator
        sim = simgen.geometrical_simulation(nickel_rlp.symmetrise())

        se = EBSD(np.ones(simgen.navigation_shape + (60, 60), dtype=np.uint8))

        with pytest.warns(UserWarning, match="Matplotlib will print"):
            se.add_marker(
                marker=sim.zone_axes_labels_as_markers(),
                permanent=False,
                plot_marker=True,
            )
            se.plot()
            se.axes_manager[0].index = 1
        plt.close("all")

    @pytest.mark.parametrize("nav_shape", [(5, 5), (1, 25), (25, 1), (25,)])
    def test_pc_as_markers(
        self, nickel_ebsd_simulation_generator, nickel_rlp, nav_shape
    ):
        """Projection center markers work."""
        simgen = nickel_ebsd_simulation_generator
        simgen.navigation_shape = nav_shape
        sim = simgen.geometrical_simulation(nickel_rlp.symmetrise())

        se = EBSD(np.ones(nav_shape + (60, 60), dtype=np.uint8))

        se.add_marker(
            marker=sim.pc_as_markers(), permanent=True, plot_marker=False
        )
        assert isinstance(se.metadata.Markers.point, point)
        assert se.metadata.Markers.point.marker_properties["marker"] == "*"
        se.plot()
        plt.close("all")

    def test_as_markers(self, nickel_ebsd_simulation_generator, nickel_rlp):
        """All markers work."""
        simgen = nickel_ebsd_simulation_generator
        sim = simgen.geometrical_simulation(nickel_rlp.symmetrise())

        se = EBSD(np.ones(simgen.navigation_shape + (60, 60), dtype=np.uint8))

        matplotlib.set_loglevel("error")
        se.add_marker(
            marker=sim.as_markers(
                bands=True, zone_axes=True, zone_axes_labels=True, pc=True,
            ),
            permanent=True,
            plot_marker=False,
        )
        se.plot()
        plt.close("all")
        matplotlib.set_loglevel("warning")

    def test_zone_axes_within_gnomonic_bounds(
        self, nickel_ebsd_simulation_generator, nickel_rlp
    ):
        """Desired boolean array."""
        simgen = nickel_ebsd_simulation_generator
        simgen.navigation_shape = (5, 5)
        rlp = nickel_rlp.symmetrise()

        # 0d: 6 bands
        simgen0d = simgen[0, 0]
        simgen0d.navigation_shape = ()
        sim0d = simgen0d.geometrical_simulation(rlp[:10])
        assert sim0d.zone_axes_within_gnomonic_bounds.shape == (6,)

        # 1d: 3 points, 7 bands
        sim1d = simgen[0, :3].geometrical_simulation(rlp[:10])
        assert sim1d.zone_axes_within_gnomonic_bounds.shape == (3, 7)

        # 2d, 4 by 2 points, 7 bands
        sim2d = simgen[:4, :2].geometrical_simulation(rlp[:10])
        assert sim2d.zone_axes_within_gnomonic_bounds.shape == (4, 2, 7)

    @pytest.mark.parametrize("nav_shape", [(5, 5), (25, 1), (1, 25)])
    def test_repr(
        self, nickel_ebsd_simulation_generator, nickel_rlp, nav_shape
    ):
        """Desired string representation."""
        simgen = nickel_ebsd_simulation_generator
        simgen.navigation_shape = nav_shape
        sim = simgen.geometrical_simulation(nickel_rlp)

        assert repr(sim) == (
            f"GeometricalEBSDSimulation {nav_shape}\n"
            "EBSDDetector (60, 60), px_size 70 um, binning 8, tilt 0, pc "
            "(0.421, 0.221, 0.505)\n"
            "<name: ni. space group: Fm-3m. point group: m-3m. proper point "
            "group: 432. color: tab:blue>\n"
            f"KikuchiBand {str(nav_shape)[:-1]}|3)\n"
            f"Rotation {nav_shape}"
        )

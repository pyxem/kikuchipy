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

import matplotlib.colors as mcolors
import matplotlib.figure as mfigure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from packaging.version import Version
import pytest

import kikuchipy as kp
from kikuchipy._constants import dependency_version


class TestEBSDDetectorPlot:
    @pytest.mark.parametrize(
        "coordinates, show_pc, pattern, zoom, desired_label",
        [
            (None, False, None, 1, "detector"),
            ("detector", True, np.ones((60, 60)), 1, "detector"),
            ("gnomonic", True, np.ones((60, 60)), 2, "gnomonic"),
        ],
    )
    def test_plot_detector(
        self, detector, coordinates, show_pc, pattern, zoom, desired_label
    ):
        """Plotting detector works, *not* checking whether Matplotlib
        displays the pattern correctly.
        """
        kwargs = {
            "show_pc": show_pc,
            "pattern": pattern,
            "zoom": zoom,
            "return_figure": True,
        }
        if coordinates is not None:
            kwargs["coordinates"] = coordinates
        fig = detector.plot(**kwargs)
        ax = fig.axes[0]
        assert ax.get_xlabel() == f"{desired_label.capitalize()} X"
        assert ax.get_ylabel() == f"{desired_label.capitalize()} Y"
        if isinstance(pattern, np.ndarray):
            assert np.allclose(ax.get_images()[0].get_array(), pattern)
        plt.close("all")

    @pytest.mark.parametrize(
        "gnomonic_angles, gnomonic_circles_kwargs",
        [
            ([10, 20], {"edgecolor": "b"}),
            (np.arange(1, 3) * 10, {"edgecolor": "r"}),
            (None, None),
        ],
    )
    def test_plot_detector_gnomonic_circles(
        self, detector, gnomonic_angles, gnomonic_circles_kwargs
    ):
        """Draw gnomonic circles."""
        fig = detector.plot(
            coordinates="gnomonic",
            draw_gnomonic_circles=True,
            gnomonic_angles=gnomonic_angles,
            gnomonic_circles_kwargs=gnomonic_circles_kwargs,
            return_figure=True,
        )
        ax = fig.axes[0]

        # Correct number of gnomonic circles are added to the patches
        num_circles = 0
        num_rectangles = 0
        for patch in ax.patches:
            if isinstance(patch, mpatches.Circle):
                num_circles += 1
            elif isinstance(patch, mpatches.Rectangle):
                num_rectangles += 1

        if gnomonic_angles is None:
            assert num_circles == 8  # Default
        else:
            assert num_circles == len(gnomonic_angles)
        assert num_rectangles == 1

        # Circles are coloured correctly
        if gnomonic_circles_kwargs is None:
            edgecolor = "k"
        else:
            edgecolor = gnomonic_circles_kwargs["edgecolor"]
        assert np.allclose(ax.patches[1]._edgecolor[:3], mcolors.to_rgb(edgecolor))

        plt.close("all")

    @pytest.mark.parametrize("pattern", [np.ones((61, 61)), np.ones((59, 60))])
    def test_plot_detector_pattern_raises(self, detector, pattern):
        """Pattern shape unequal to detector shape raises ValueError."""
        with pytest.raises(ValueError, match=f"Pattern shape {pattern.shape}*"):
            detector.plot(pattern=pattern)
        plt.close("all")

    @pytest.mark.parametrize(
        "pattern_kwargs", [None, {"cmap": "inferno"}, {"cmap": "plasma"}]
    )
    def test_plot_pattern_kwargs(self, detector, pattern_kwargs):
        """Pass pattern kwargs to imshow()."""
        fig = detector.plot(
            pattern=np.ones((60, 60)),
            pattern_kwargs=pattern_kwargs,
            return_figure=True,
        )
        if pattern_kwargs is None:
            pattern_kwargs = {"cmap": "gray"}
        assert fig.axes[0].images[0].cmap.name == pattern_kwargs["cmap"]
        plt.close("all")

    def test_plot_pc_style(self, detector):
        fig = detector.plot(show_pc=True, return_figure=True)
        colls = fig.axes[0].collections

        # White circle with a black cross
        assert len(colls) == 2
        assert np.allclose(colls[0].get_facecolor(), mcolors.to_rgba("w"))
        assert np.allclose(colls[1].get_facecolor(), mcolors.to_rgba("k"))

        plt.close("all")

    @pytest.mark.parametrize("coordinates", ["detector", "gnomonic"])
    def test_plot_extent(self, detector, coordinates):
        """Correct detector extent."""
        fig = detector.plot(
            coordinates=coordinates,
            pattern=np.ones(detector.shape),
            return_figure=True,
        )
        if coordinates == "gnomonic":
            desired_data_lim = np.concatenate(
                [
                    detector._average_gnomonic_bounds[::2],
                    np.diff(detector._average_gnomonic_bounds)[::2],
                ]
            )
        else:
            desired_data_lim = np.sort(detector.bounds)
        assert np.allclose(fig.axes[0].dataLim.bounds, desired_data_lim)
        plt.close("all")


@pytest.mark.skipif(
    dependency_version["ipywidgets"] is None, reason="ipywidgets is not installed"
)
class TestPlotEBSDDetectorViews:
    def test_plot_side_view(self):
        det = kp.detectors.EBSDDetector()

        fig1 = det.plot_side_view(return_figure=True)
        ax1 = fig1.axes[0]
        line_labels = [line.get_label() for line in ax1.lines]
        assert all(expected in line_labels for expected in ["Sample", "Detector"])
        coll_labels = [coll.get_label() for coll in ax1.collections]
        assert all(label in coll_labels for label in ["_pc_circle", "_pc_cross"])
        assert len(ax1.texts) == 1
        assert "Microscope Y" in ax1.get_xlabel()
        assert "Microscope Z" in ax1.get_ylabel()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot()
        det.plot_side_view(ax=ax2, legend=True, dimensionless=False)
        legend = fig2.legend()
        if dependency_version["matplotlib"] >= Version("3.7"):
            assert len(legend.legend_handles) == 2
        else:
            assert len(legend.legendHandles) == 2
        assert "[mm]" in ax2.get_xlabel()
        assert "[mm]" in ax2.get_ylabel()

    def test_plot_side_view_interactive(self):
        import ipywidgets

        det = kp.detectors.EBSDDetector()
        widgets1 = det.plot_side_view(interactive=True)
        assert isinstance(widgets1, ipywidgets.VBox)

        widgets2, fig1 = det.plot_side_view(interactive=True, return_figure=True)
        assert isinstance(widgets2, ipywidgets.VBox)
        assert isinstance(fig1, mfigure.Figure)

        _, ax = plt.subplots()
        _, fig3 = det.plot_side_view(interactive=True, ax=ax, return_figure=True)
        assert fig3.axes[0] is ax


class TestPlotPC:
    det = kp.detectors.EBSDDetector(
        shape=(60, 60),
        pc=np.stack(
            (
                np.repeat(np.linspace(0.55, 0.45, 30), 20).reshape(30, 20).T,
                np.repeat(np.linspace(0.75, 0.70, 20), 30).reshape(20, 30),
                np.repeat(np.linspace(0.50, 0.55, 20), 30).reshape(20, 30),
            ),
            axis=2,
        ),
        sample_tilt=70,
    )

    def test_plot_pc_raises(self):
        det = self.det.deepcopy()
        det.pc = det.pc_average
        with pytest.raises(ValueError, match="Detector must have more than one "):
            det.plot_pc()

        det2 = self.det.deepcopy()
        det2.pc = det2.pc[0]
        with pytest.raises(ValueError, match="Detector's navigation dimension must be"):
            det2.plot_pc()

        with pytest.raises(ValueError, match="Plot mode 'stereographic' must be one "):
            self.det.plot_pc("stereographic")

    def test_plot_pc_map_horizontal(self):
        fig = self.det.plot_pc(return_figure=True)

        figsize = fig.get_size_inches()
        assert (figsize[0] / figsize[1]) > 1

        axes = fig.axes
        assert len(axes) == 6
        assert all([ax.get_xlabel() == "Column" for ax in axes[:3]])
        for ax, label in zip(axes[3:], ["x", "y", "z"]):
            assert ax.get_ylabel() == f"PC{label}"

        plt.close(fig)

    def test_plot_pc_map_vertical(self):
        fig = self.det.plot_pc(return_figure=True, orientation="vertical")

        figsize = fig.get_size_inches()
        assert (figsize[0] / figsize[1]) < 1

        axes = fig.axes
        assert len(axes) == 6
        assert all([ax.get_xlabel() == "Column" for ax in axes[:3]])
        for ax, label in zip(axes[3:], ["x", "y", "z"]):
            assert ax.get_ylabel() == f"PC{label}"

        plt.close(fig)

    def test_plot_pc_scatter_horizontal(self):
        fig = self.det.plot_pc("scatter", return_figure=True, annotate=True)

        figsize = fig.get_size_inches()
        assert (figsize[0] / figsize[1]) > 1

        axes = fig.axes
        assert len(axes) == 3
        for ax, label in zip(axes[3:], ["x", "y", "z"]):
            assert ax.get_xlabel() == f"PC{label}"
            assert ax.get_ylabel() == f"PC{label}"

        texts = axes[0].texts
        assert len(texts) == self.det.navigation_size
        assert texts[0].get_text() == "0"
        assert texts[-1].get_text() == "599"

        plt.close(fig)

    def test_plot_pc_scatter_vertical(self):
        fig = self.det.plot_pc("scatter", return_figure=True, orientation="vertical")

        figsize = fig.get_size_inches()
        assert (figsize[0] / figsize[1]) < 1

        axes = fig.axes
        assert len(axes) == 3
        for ax, label in zip(axes[3:], ["x", "y", "z"]):
            assert ax.get_xlabel() == f"PC{label}"
            assert ax.get_ylabel() == f"PC{label}"

        plt.close(fig)

    def test_plot_pc_3d(self):
        fig = self.det.plot_pc("3d", return_figure=True, annotate=True)

        texts = fig.axes[0].texts
        assert len(texts) == self.det.navigation_size
        assert texts[0].get_text() == "0"
        assert texts[-1].get_text() == "599"

        plt.close(fig)

    def test_plot_pc_figure(self):
        fig1 = self.det.plot_pc(figure_kwargs=dict(figsize=(9, 3)), return_figure=True)
        assert fig1.get_tight_layout()

        fig2 = self.det.plot_pc(
            figure_kwargs=dict(figsize=(6, 3), layout="constrained"), return_figure=True
        )
        assert fig2.get_constrained_layout()
        assert not np.allclose(fig1.get_size_inches(), fig2.get_size_inches())

        plt.close("all")

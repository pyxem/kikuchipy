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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from packaging.version import Version
import pytest

import kikuchipy as kp
from kikuchipy._constants import dependency_version


def _get_patch_by_label(ax, label):
    for patch in ax.patches:
        if patch.get_label() == label:
            return patch
    return None


def _get_polygon_corners(patch):
    return np.asarray(patch.get_xy())[:4]


def _get_collection_by_label(ax, label):
    for coll in ax.collections:
        if coll.get_label() == label:
            return coll
    return None


class TestEBSDDetectorPlot:
    @pytest.mark.parametrize(
        "coordinates, show_pc, pattern, zoom, desired_label",
        [
            (None, False, None, 1, "detector"),
            ("pixel", True, np.ones((60, 60)), 1, "detector"),
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

    @pytest.mark.parametrize("coordinates", ["pixel", "gnomonic"])
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

    def test_plot_top_view(self):
        det = kp.detectors.EBSDDetector()

        fig1 = det.plot_top_view(return_figure=True)
        ax1 = fig1.axes[0]
        patch_labels = [patch.get_label() for patch in ax1.patches]
        assert all(expected in patch_labels for expected in ["Sample", "Detector"])
        coll_labels = [coll.get_label() for coll in ax1.collections]
        assert all(label in coll_labels for label in ["_pc_circle", "_pc_cross"])
        assert "Microscope X" in ax1.get_xlabel()
        assert "Microscope Y" in ax1.get_ylabel()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot()
        det.plot_top_view(ax=ax2, legend=True, dimensionless=False)
        legend = fig2.legend()
        if dependency_version["matplotlib"] >= Version("3.7"):
            assert len(legend.legend_handles) == 3
        else:
            assert len(legend.legendHandles) == 3
        assert "[mm]" in ax2.get_xlabel()
        assert "[mm]" in ax2.get_ylabel()

    def test_plot_top_view_outline_style(self):
        det = kp.detectors.EBSDDetector()

        fig = det.plot_top_view(return_figure=True)
        ax = fig.axes[0]
        sample_patch = _get_patch_by_label(ax, "Sample")
        detector_patch = _get_patch_by_label(ax, "Detector")
        assert sample_patch is not None
        assert detector_patch is not None

        for patch, facecolor in [
            (sample_patch, "C0"),
            (detector_patch, "C7"),
        ]:
            edge = np.asarray(patch.get_edgecolor())
            if edge.ndim == 2:
                edge = edge[0]
            assert np.allclose(edge, mcolors.to_rgba("k", alpha=0.5))

            face = np.asarray(patch.get_facecolor())
            if face.ndim == 2:
                face = face[0]
            assert np.allclose(face, mcolors.to_rgba(facecolor, alpha=0.5))
            assert np.isclose(patch.get_linewidth(), 1.5)

        plt.close(fig)

    def test_plot_top_view_sample_projection_with_sample_tilt(self):
        det = kp.detectors.EBSDDetector(shape=(80, 40), sample_tilt=90, tilt=0)

        fig1 = det.plot_top_view(return_figure=True)
        sample_patch1 = _get_patch_by_label(fig1.axes[0], "Sample")
        assert sample_patch1 is not None
        sample_xy_90 = np.asarray(sample_patch1.get_xy())
        assert np.isclose(np.ptp(sample_xy_90[:, 1]), 0, atol=1e-12)
        plt.close(fig1)

        det.sample_tilt = 0
        fig2 = det.plot_top_view(return_figure=True)
        sample_patch2 = _get_patch_by_label(fig2.axes[0], "Sample")
        assert sample_patch2 is not None
        sample_xy_0 = np.asarray(sample_patch2.get_xy())
        sample_size_mm = 0.6 * det.height * 1e-3
        assert np.isclose(np.ptp(sample_xy_0[:, 0]), sample_size_mm)
        assert np.isclose(np.ptp(sample_xy_0[:, 1]), sample_size_mm)
        plt.close(fig2)

    @pytest.mark.parametrize("tilt", [-90, 90])
    def test_plot_top_view_detector_projection_at_extreme_tilt(self, tilt):
        det = kp.detectors.EBSDDetector(
            shape=(120, 60),
            sample_tilt=90,
            tilt=tilt,
            azimuthal=0,
            pc=[0.5, 0.5, 0.2],
        )

        fig = det.plot_top_view(return_figure=True)
        detector_patch = _get_patch_by_label(fig.axes[0], "Detector")
        assert detector_patch is not None
        detector_xy = np.asarray(detector_patch.get_xy())
        assert np.isclose(np.ptp(detector_xy[:, 0]), det.width * 1e-3)
        assert np.isclose(np.ptp(detector_xy[:, 1]), det.height * 1e-3)
        plt.close(fig)

    def test_plot_top_view_detector_center_moves_with_detector_tilt(self):
        det = kp.detectors.EBSDDetector(
            shape=(120, 60),
            sample_tilt=90,
            tilt=0,
            azimuthal=0,
            pc=[0.5, 0.5, 0.5],
        )

        fig1 = det.plot_top_view(return_figure=True)
        det_patch1 = _get_patch_by_label(fig1.axes[0], "Detector")
        assert det_patch1 is not None
        det_xy_0 = _get_polygon_corners(det_patch1)
        y_center_0 = float(0.5 * (np.max(det_xy_0[:, 1]) + np.min(det_xy_0[:, 1])))
        plt.close(fig1)

        det.tilt = 90
        fig2 = det.plot_top_view(return_figure=True)
        det_patch2 = _get_patch_by_label(fig2.axes[0], "Detector")
        assert det_patch2 is not None
        det_xy_90 = _get_polygon_corners(det_patch2)
        y_center_90 = float(0.5 * (np.max(det_xy_90[:, 1]) + np.min(det_xy_90[:, 1])))
        plt.close(fig2)

        assert np.isclose(y_center_0, det.specimen_scintillator_distance[0] * 1e-3)
        assert np.isclose(y_center_90, 0, atol=1e-12)

    @pytest.mark.parametrize(
        "detector_tilt",
        [
            90,
            -90,
            0,
        ],
    )
    def test_plot_top_view_always_shows_sample_and_detector(self, detector_tilt):
        det = kp.detectors.EBSDDetector(
            shape=(60, 60),
            sample_tilt=90,
            tilt=detector_tilt,
            azimuthal=0,
            pc=[0.5, 0.5, 0.5],
        )

        fig = det.plot_top_view(return_figure=True)
        patch_labels = {patch.get_label() for patch in fig.axes[0].patches}
        plt.close(fig)

        assert {"Sample", "Detector"}.issubset(patch_labels)

    @pytest.mark.parametrize("detector_tilt", [-90, 90])
    def test_plot_top_view_azimuthal_at_extreme_tilt_is_not_twist(self, detector_tilt):
        det = kp.detectors.EBSDDetector(
            shape=(120, 60),
            sample_tilt=90,
            tilt=detector_tilt,
            azimuthal=0,
            pc=[0.5, 0.5, 0.5],
        )

        fig0 = det.plot_top_view(return_figure=True)
        patch0 = _get_patch_by_label(fig0.axes[0], "Detector")
        assert patch0 is not None
        corners0 = _get_polygon_corners(patch0)
        center0 = corners0.mean(axis=0)
        plt.close(fig0)

        det.azimuthal = 45
        fig45 = det.plot_top_view(return_figure=True)
        patch45 = _get_patch_by_label(fig45.axes[0], "Detector")
        assert patch45 is not None
        corners45 = _get_polygon_corners(patch45)
        center45 = corners45.mean(axis=0)
        plt.close(fig45)

        # Azimuthal at extreme detector tilt should tilt the detector
        # about detector Y, changing projected center/width, not produce
        # an in-plane twist.
        assert not np.isclose(center45[0], center0[0])
        expected_x_span = det.width * np.cos(np.deg2rad(det.azimuthal)) * 1e-3
        assert np.isclose(np.ptp(corners45[:, 0]), expected_x_span)
        assert np.isclose(np.ptp(corners45[:, 1]), det.height * 1e-3)

    @pytest.mark.parametrize(
        "detector_tilt, sample_on_top",
        [
            (20, True),
            (-20, False),
        ],
    )
    def test_plot_top_view_zorder_follows_detector_tilt(
        self, detector_tilt, sample_on_top
    ):
        det = kp.detectors.EBSDDetector(
            shape=(60, 60),
            sample_tilt=70,
            tilt=detector_tilt,
            azimuthal=10,
            pc=[0.5, 0.5, 0.5],
        )

        fig = det.plot_top_view(return_figure=True)
        ax = fig.axes[0]
        sample_patch = _get_patch_by_label(ax, "Sample")
        detector_patch = _get_patch_by_label(ax, "Detector")
        assert sample_patch is not None
        assert detector_patch is not None

        if sample_on_top:
            assert sample_patch.get_zorder() > detector_patch.get_zorder()
        else:
            assert detector_patch.get_zorder() > sample_patch.get_zorder()

        plt.close(fig)

    @pytest.mark.parametrize(
        "detector_tilt, beam_above_detector",
        [
            (20, True),
            (-20, False),
        ],
    )
    def test_plot_top_view_beam_zorder_follows_detector_tilt(
        self, detector_tilt, beam_above_detector
    ):
        det = kp.detectors.EBSDDetector(
            shape=(60, 60),
            sample_tilt=70,
            tilt=detector_tilt,
            azimuthal=10,
            pc=[0.5, 0.5, 0.5],
        )

        fig = det.plot_top_view(return_figure=True)
        ax = fig.axes[0]
        sample_patch = _get_patch_by_label(ax, "Sample")
        detector_patch = _get_patch_by_label(ax, "Detector")
        beam_coll = _get_collection_by_label(ax, "Beam")
        assert sample_patch is not None
        assert detector_patch is not None
        assert beam_coll is not None

        assert beam_coll.get_zorder() > sample_patch.get_zorder()
        if beam_above_detector:
            assert beam_coll.get_zorder() > detector_patch.get_zorder()
        else:
            assert beam_coll.get_zorder() < detector_patch.get_zorder()

        plt.close(fig)


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
        for ax, (label_x, label_y) in zip(axes, [("x", "y"), ("x", "z"), ("z", "y")]):
            assert ax.get_xlabel() == f"PC{label_x}"
            assert ax.get_ylabel() == f"PC{label_y}"

        texts = axes[0].texts
        assert len(texts) == self.det.navigation_size
        assert texts[0].get_text() == "0"
        assert texts[-1].get_text() == "599"

        plt.close(fig)

    def test_plot_pc_scatter_vertical(self):
        fig = self.det.plot_pc("scatter", return_figure=True, orientation="vertical")

        figsize = fig.get_size_inches()
        assert (figsize[0] / figsize[1]) < 1

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

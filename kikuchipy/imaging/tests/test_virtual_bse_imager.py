# Copyright 2019-2023 The kikuchipy developers
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

import os

from hyperspy.roi import RectangularROI
import numpy as np
import pytest

import kikuchipy as kp


DIR_PATH = os.path.dirname(__file__)
KIKUCHIPY_FILE = os.path.join(DIR_PATH, "../../data/kikuchipy_h5ebsd/patterns.h5")


class TestVirtualBSEImager:
    def test_init(self, dummy_signal):
        vbse_imager = kp.imaging.VirtualBSEImager(dummy_signal)

        assert isinstance(vbse_imager.signal, kp.signals.EBSD)
        assert vbse_imager.grid_shape == (5, 5)

    def test_init_lazy(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()
        vbse_imager = kp.imaging.VirtualBSEImager(lazy_signal)

        assert isinstance(vbse_imager.signal, kp.signals.LazyEBSD)

    @pytest.mark.parametrize(
        "grid_shape, desired_rows, desired_cols",
        [
            (
                (10, 10),
                np.linspace(0, 60, 10 + 1),
                np.linspace(0, 60, 10 + 1),
            ),
            (
                (13, 7),
                np.linspace(0, 60, 13 + 1),
                np.linspace(0, 60, 7 + 1),
            ),
        ],
    )
    def test_set_grid_shape(self, grid_shape, desired_rows, desired_cols):
        s = kp.load(KIKUCHIPY_FILE)
        vbse_imager = kp.imaging.VirtualBSEImager(s)
        vbse_imager.grid_shape = grid_shape

        assert vbse_imager.grid_shape == grid_shape
        assert np.allclose(vbse_imager.grid_rows, desired_rows)
        assert np.allclose(vbse_imager.grid_cols, desired_cols)

    def test_repr(self, dummy_signal):
        vbse_imager = kp.imaging.VirtualBSEImager(dummy_signal)

        assert repr(vbse_imager) == (
            "VirtualBSEImager for <EBSD, title: , dimensions: (3, 3|3, 3)>"
        )

    @pytest.mark.parametrize(
        "grid_shape, desired_n_markers",
        [((3, 3), 1 + 1 + 3 + 9), ((1, 1), 1 + 1 + 3 + 1), ((2, 3), 1 + 1 + 3 + 6)],
    )
    def test_plot_grid(self, grid_shape, desired_n_markers):
        s = kp.load(KIKUCHIPY_FILE)
        vbse_imager = kp.imaging.VirtualBSEImager(s)
        vbse_imager.grid_shape = grid_shape
        rgb_channels = [(0, 0), (0, 1), (1, 0)]
        pattern_idx = (2, 2)
        p = vbse_imager.plot_grid(pattern_idx=pattern_idx, rgb_channels=rgb_channels)
        p2 = vbse_imager.plot_grid()

        # Check data type and values
        assert isinstance(p, kp.signals.EBSD)
        assert np.allclose(p.data, s.inav[pattern_idx].data)
        assert np.allclose(p2.data, s.inav[0, 0].data)

        # Check markers
        assert len(p.metadata.Markers) == desired_n_markers
        assert p.metadata.Markers.has_item("Texts")
        assert p.metadata.Markers.Texts.kwargs["color"] == ("r",)
        assert p.metadata.Markers.HorizontalLines.kwargs["color"] == ("w",)
        assert p.metadata.Markers.Rectangles.kwargs["ec"] == "r"

    @pytest.mark.parametrize("color", ["c", "m", "k"])
    def test_plot_grid_text_color(self, color):
        s = kp.load(KIKUCHIPY_FILE)
        vbse_imager = kp.imaging.VirtualBSEImager(s)
        p = vbse_imager.plot_grid(color=color)
        assert p.metadata.Markers["Texts"].kwargs["color"] == (color,)


class TestGetImagesFromGrid:
    def test_get_single_image_from_grid(self, dummy_signal):
        vbse_imager = kp.imaging.VirtualBSEImager(dummy_signal)
        vbse_imager.grid_shape = (1, 1)
        vbse_img = vbse_imager.get_images_from_grid()

        assert np.allclose(vbse_img.data.mean(), 40.666668)

    @pytest.mark.parametrize("dtype_out", [np.float32, np.float64])
    def test_dtype_out(self, dummy_signal, dtype_out):
        vbse_imager = kp.imaging.VirtualBSEImager(dummy_signal)
        vbse_imager.grid_shape = (1, 1)
        vbse_images = vbse_imager.get_images_from_grid(dtype_out=dtype_out)

        assert vbse_images.data.dtype == dtype_out

    def test_axes_manager_transfer(self):
        s = kp.load(KIKUCHIPY_FILE)
        vbse_imager = kp.imaging.VirtualBSEImager(s)
        vbse_img = vbse_imager.get_images_from_grid()

        s_nav_axes = s.axes_manager.navigation_axes
        vbse_sig_axes = vbse_img.axes_manager.signal_axes

        assert all([vbse_sig_axes[i].scale == s_nav_axes[i].scale for i in range(2)])
        assert all([vbse_sig_axes[i].name == s_nav_axes[i].name for i in range(2)])
        assert all([vbse_sig_axes[i].units == s_nav_axes[i].units for i in range(2)])

    def test_get_images_lazy(self):
        s = kp.load(KIKUCHIPY_FILE, lazy=True)
        vbse_imager = kp.imaging.VirtualBSEImager(s)
        _ = vbse_imager.get_images_from_grid()


class TestGetRGBImage:
    def test_get_rgb_image_rois(self):
        s = kp.load(KIKUCHIPY_FILE)
        vbse_imager = kp.imaging.VirtualBSEImager(s)

        # Get channels by ROIs
        rois1 = [
            vbse_imager.roi_from_grid(i) for i in np.ndindex(vbse_imager.grid_shape)
        ][:3]
        vbse_rgb_img1 = vbse_imager.get_rgb_image(r=rois1[0], g=rois1[1], b=rois1[2])

        # Get channels from grid tile indices
        rois2 = [(0, 0), (0, 1), (0, 2)]
        vbse_rgb_img2 = vbse_imager.get_rgb_image(r=rois2[0], g=rois2[1], b=rois2[2])

        assert isinstance(vbse_rgb_img1, kp.signals.VirtualBSEImage)
        assert vbse_rgb_img1.data.dtype == np.dtype(
            [("R", "u1"), ("G", "u1"), ("B", "u1")]
        )

        vbse_rgb_img1.change_dtype("uint8")
        vbse_rgb_img2.change_dtype("uint8")
        assert np.allclose(vbse_rgb_img1.data, vbse_rgb_img2.data)

    def test_get_rgb_image_dtype(self):
        s = kp.load(KIKUCHIPY_FILE)
        vbse_imager = kp.imaging.VirtualBSEImager(s)
        vbse_rgb_img = vbse_imager.get_rgb_image(
            r=(0, 0), g=(0, 1), b=(0, 2), dtype_out=np.uint16
        )

        assert vbse_rgb_img.data.dtype == np.dtype(
            [("R", "u2"), ("G", "u2"), ("B", "u2")]
        )

    @pytest.mark.parametrize(
        "percentile, desired_mean_intensity",
        [(None, 136.481481), ((1, 99), 134.740740)],
    )
    def test_get_rgb_image_contrast_stretching(
        self, percentile, desired_mean_intensity
    ):
        s = kp.load(KIKUCHIPY_FILE)
        vbse_imager = kp.imaging.VirtualBSEImager(s)
        vbse_rgb_img = vbse_imager.get_rgb_image(
            r=(0, 0), g=(0, 1), b=(0, 2), percentiles=percentile
        )
        vbse_rgb_img.change_dtype("uint8")

        assert np.allclose(vbse_rgb_img.data.mean(), desired_mean_intensity)

    @pytest.mark.parametrize(
        "alpha_add, desired_mean_intensity", [(0, 88.5), (10, 107.9)]
    )
    def test_get_rgb_alpha(self, alpha_add, desired_mean_intensity):
        s = kp.load(KIKUCHIPY_FILE)
        vbse_imager = kp.imaging.VirtualBSEImager(s)

        alpha = np.arange(9).reshape((3, 3))
        alpha[0] += alpha_add

        vbse_rgb_img = vbse_imager.get_rgb_image(
            r=(0, 0), g=(0, 1), b=(0, 2), alpha=alpha
        )
        vbse_rgb_img.change_dtype("uint8")

        assert np.allclose(vbse_rgb_img.data.mean(), desired_mean_intensity, atol=0.1)

    def test_get_rgb_alpha_signal(self):
        s = kp.load(KIKUCHIPY_FILE)
        vbse_imager = kp.imaging.VirtualBSEImager(s)

        vbse_img = s.get_virtual_bse_intensity(roi=RectangularROI(0, 0, 10, 10))

        vbse_rgb_img1 = vbse_imager.get_rgb_image(
            r=(0, 1), g=(0, 2), b=(0, 3), alpha=vbse_img
        )
        vbse_rgb_img2 = vbse_imager.get_rgb_image(
            r=(0, 1), g=(0, 2), b=(0, 3), alpha=vbse_img.data
        )
        vbse_rgb_img1.change_dtype("uint8")
        vbse_rgb_img2.change_dtype("uint8")

        assert np.allclose(vbse_rgb_img1.data, vbse_rgb_img2.data)

    def test_get_rgb_image_lazy(self):
        s = kp.load(KIKUCHIPY_FILE, lazy=True)
        vbse_imager = kp.imaging.VirtualBSEImager(s)

        assert isinstance(vbse_imager.signal, kp.signals.LazyEBSD)

        vbse_rgb_img = vbse_imager.get_rgb_image(r=(0, 0), g=(0, 1), b=(0, 2))

        assert isinstance(vbse_rgb_img.data, np.ndarray)

    def test_get_rgb_1d(self):
        s = kp.signals.EBSD(np.random.random(9 * 3600).reshape((9, 60, 60)))
        vbse_imager = kp.imaging.VirtualBSEImager(s)

        with pytest.raises(ValueError, match="The signal dimension cannot be "):
            _ = vbse_imager.get_rgb_image(r=(0, 0), g=(0, 1), b=(0, 2))

    @pytest.mark.parametrize(
        "r, g, b, desired_mean_intensity",
        [
            ([(0, 1), (0, 2)], [(1, 1), (1, 2)], [(2, 1), (2, 2)], 125.1),
            ([(2, 1), (2, 2)], [(3, 1), (3, 2)], [(4, 1), (4, 2)], 109.0),
        ],
    )
    def test_get_rgb_multiple_rois_per_channel(self, r, g, b, desired_mean_intensity):
        s = kp.load(KIKUCHIPY_FILE)
        vbse_imager = kp.imaging.VirtualBSEImager(s)

        vbse_rgb_img1 = vbse_imager.get_rgb_image(r=r, g=g, b=b)
        vbse_rgb_img1.change_dtype("uint8")

        assert np.allclose(vbse_rgb_img1.data.mean(), desired_mean_intensity, atol=0.1)

        roi_r = vbse_imager.roi_from_grid(r)
        roi_g = vbse_imager.roi_from_grid(g)
        roi_b = vbse_imager.roi_from_grid(b)
        vbse_rgb_img2 = vbse_imager.get_rgb_image(r=roi_r, g=roi_g, b=roi_b)
        vbse_rgb_img2.change_dtype("uint8")

        assert np.allclose(vbse_rgb_img1.data, vbse_rgb_img2.data)

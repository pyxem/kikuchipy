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

import os

import numpy as np
import pytest

from kikuchipy import load
from kikuchipy.generators import VirtualBSEGenerator
from kikuchipy.signals import EBSD, LazyEBSD, VirtualBSEImage

DIR_PATH = os.path.dirname(__file__)
KIKUCHIPY_FILE = os.path.join(DIR_PATH, "../../data/kikuchipy/patterns.h5")


class TestVirtualBSEGenerator:
    def test_init(self, dummy_signal):
        vbse_gen = VirtualBSEGenerator(dummy_signal)

        assert isinstance(vbse_gen.signal, EBSD)
        assert vbse_gen.grid_shape == (5, 5)

    def test_init_lazy(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()
        vbse_gen = VirtualBSEGenerator(lazy_signal)

        assert isinstance(vbse_gen.signal, LazyEBSD)

    @pytest.mark.parametrize(
        "grid_shape, desired_rows, desired_cols",
        [
            ((10, 10), np.linspace(0, 60, 10 + 1), np.linspace(0, 60, 10 + 1),),
            ((13, 7), np.linspace(0, 60, 13 + 1), np.linspace(0, 60, 7 + 1),),
        ],
    )
    def test_set_grid_shape(self, grid_shape, desired_rows, desired_cols):
        s = load(KIKUCHIPY_FILE)
        vbse_gen = VirtualBSEGenerator(s)
        vbse_gen.grid_shape = grid_shape

        assert vbse_gen.grid_shape == grid_shape
        assert np.allclose(vbse_gen.grid_rows, desired_rows)
        assert np.allclose(vbse_gen.grid_cols, desired_cols)

    def test_repr(self, dummy_signal):
        vbse_gen = VirtualBSEGenerator(dummy_signal)

        assert vbse_gen.__repr__() == (
            "VirtualBSEGenerator for <EBSD, title: , dimensions: (3, 3|3, 3)>"
        )

    def test_get_images_from_grid(self):
        pass

    def test_plot_grid(self):
        pass

    def test_roi_from_grid(self):
        pass


class TestGetImagesFromGrid:
    pass


class TestGetRGBImage:
    def test_get_rgb_image_rois(self):
        s = load(KIKUCHIPY_FILE)
        vbse_gen = VirtualBSEGenerator(s)

        # Get channels by ROIs
        rois1 = [
            vbse_gen.roi_from_grid(row=r, col=c)
            for r, c in np.ndindex(vbse_gen.grid_shape)
        ][:3]
        vbse_rgb_img1 = vbse_gen.get_rgb_image(rois=rois1)

        # Get channels from grid tile indices
        rois2 = [(0, 0), (0, 1), (0, 2)]
        vbse_rgb_img2 = vbse_gen.get_rgb_image(rois=rois2)

        assert isinstance(vbse_rgb_img1, VirtualBSEImage)
        assert vbse_rgb_img1.data.dtype == np.dtype(
            [("R", "u1"), ("G", "u1"), ("B", "u1")]
        )

        vbse_rgb_img1.change_dtype(np.uint8)
        vbse_rgb_img2.change_dtype(np.uint8)
        assert np.allclose(vbse_rgb_img1.data, vbse_rgb_img2.data)

    def test_get_rgb_image_dtype(self):
        s = load(KIKUCHIPY_FILE)
        vbse_gen = VirtualBSEGenerator(s)
        vbse_rgb_img = vbse_gen.get_rgb_image(
            rois=[(0, 0), (0, 1), (0, 2)], dtype_out=np.uint16,
        )

        assert vbse_rgb_img.data.dtype == np.dtype(
            [("R", "u2"), ("G", "u2"), ("B", "u2")]
        )

    @pytest.mark.parametrize(
        "percentile, desired_mean_intensity",
        [(None, 140.14814), ((1, 99), 134.740740),],
    )
    def test_get_rgb_image_contrast_stretching(
        self, percentile, desired_mean_intensity
    ):
        s = load(KIKUCHIPY_FILE)
        vbse_gen = VirtualBSEGenerator(s)
        vbse_rgb_img = vbse_gen.get_rgb_image(
            rois=[(0, 0), (0, 1), (0, 2)], percentile=percentile,
        )
        vbse_rgb_img.change_dtype(np.uint8)

        assert np.allclose(vbse_rgb_img.data.mean(), desired_mean_intensity)

    @pytest.mark.parametrize(
        "alpha_add, desired_mean_intensity", [(0, 88.481481), (10, 59.703703),]
    )
    def test_get_rgb_alpha(self, alpha_add, desired_mean_intensity):
        s = load(KIKUCHIPY_FILE)
        vbse_gen = VirtualBSEGenerator(s)

        alpha = np.arange(9).reshape((3, 3))
        alpha[0] += alpha_add

        vbse_rgb_img = vbse_gen.get_rgb_image(
            rois=[(0, 0), (0, 1), (0, 2)], alpha=alpha
        )
        vbse_rgb_img.change_dtype(np.uint8)

        assert np.allclose(vbse_rgb_img.data.mean(), desired_mean_intensity)

    def test_get_rgb_image_lazy(self):
        s = load(KIKUCHIPY_FILE, lazy=True)
        vbse_gen = VirtualBSEGenerator(s)

        assert isinstance(vbse_gen.signal, LazyEBSD)

        vbse_rgb_img = vbse_gen.get_rgb_image(rois=[(0, 0), (0, 1), (0, 2)])

        assert isinstance(vbse_rgb_img.data, np.ndarray)

    def test_get_rgb_1d(self):
        s = EBSD(np.random.random(9 * 3600).reshape((9, 60, 60)))
        vbse_gen = VirtualBSEGenerator(s)

        with pytest.raises(ValueError, match="The signal dimension cannot be "):
            _ = vbse_gen.get_rgb_image(rois=[(0, 0), (0, 1), (0, 2)])

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

import numpy as np
import pytest

import hyperspy.api as hs
import kikuchipy as kp


class TestVirtualBSEImage:
    def test_rescale_rgb_raises(self, dummy_signal):
        vbse_gen = kp.imaging.VirtualBSEImager(dummy_signal)
        vbse_gen.grid_shape = (3, 3)
        rgb = vbse_gen.get_rgb_image(r=(0, 0), g=(0, 1), b=(1, 0))
        with pytest.raises(NotImplementedError):
            rgb.rescale_intensity()

    def test_normalize_rgb_raises(self, dummy_signal):
        vbse_gen = kp.imaging.VirtualBSEImager(dummy_signal)
        vbse_gen.grid_shape = (3, 3)
        rgb = vbse_gen.get_rgb_image(r=(0, 0), g=(0, 1), b=(1, 0))
        with pytest.raises(NotImplementedError):
            rgb.normalize_intensity()

    def test_rescale_intensity_inplace(self, dummy_signal):
        vbse_gen = kp.imaging.VirtualBSEImager(dummy_signal)
        vbse_gen.grid_shape = (3, 3)
        vbse_img = vbse_gen.get_images_from_grid()

        # Current signal is unaffected
        vbse_img2 = vbse_img.deepcopy()
        vbse_img3 = vbse_img.normalize_intensity(inplace=False)
        assert isinstance(vbse_img3, kp.signals.VirtualBSEImage)
        assert np.allclose(vbse_img2.data, vbse_img.data)

        # Operating on current signal gives same result as output
        vbse_img.normalize_intensity()
        assert np.allclose(vbse_img3.data, vbse_img.data)

        # Operating on lazy signal returns lazy signal
        mp4 = vbse_img2.as_lazy()
        mp5 = mp4.normalize_intensity(inplace=False)
        assert isinstance(mp5, kp.signals.LazyVirtualBSEImage)
        mp5.compute()
        assert np.allclose(mp5.data, vbse_img.data)

    def test_rescale_intensity_lazy_output(self, dummy_signal):
        vbse_gen = kp.imaging.VirtualBSEImager(dummy_signal)
        vbse_gen.grid_shape = (3, 3)
        vbse_img = vbse_gen.get_images_from_grid()

        with pytest.raises(
            ValueError, match="`lazy_output=True` requires `inplace=False`"
        ):
            _ = vbse_img.normalize_intensity(lazy_output=True)

        vbse_img2 = vbse_img.normalize_intensity(inplace=False, lazy_output=True)
        assert isinstance(vbse_img2, kp.signals.LazyVirtualBSEImage)

        vbse_img3 = vbse_img.as_lazy()
        vbse_img4 = vbse_img3.normalize_intensity(inplace=False, lazy_output=False)
        assert isinstance(vbse_img4, kp.signals.VirtualBSEImage)

    def test_normalize_intensity_inplace(self, dummy_signal):
        vbse_gen = kp.imaging.VirtualBSEImager(dummy_signal)
        vbse_gen.grid_shape = (3, 3)
        vbse_img = vbse_gen.get_images_from_grid()

        # Current signal is unaffected
        vbse_img2 = vbse_img.deepcopy()
        vbse_img3 = vbse_img.normalize_intensity(inplace=False)
        assert isinstance(vbse_img3, kp.signals.VirtualBSEImage)
        assert np.allclose(vbse_img2.data, vbse_img.data)

        # Operating on current signal gives same result as output
        vbse_img.normalize_intensity()
        assert np.allclose(vbse_img3.data, vbse_img.data)

        # Operating on lazy signal returns lazy signal
        vbse_img4 = vbse_img2.as_lazy()
        vbse_img5 = vbse_img4.normalize_intensity(inplace=False)
        assert isinstance(vbse_img5, kp.signals.LazyVirtualBSEImage)
        vbse_img5.compute()
        assert np.allclose(vbse_img5.data, vbse_img.data)

    def test_normalize_intensity_lazy_output(self, dummy_signal):
        vbse_gen = kp.imaging.VirtualBSEImager(dummy_signal)
        vbse_gen.grid_shape = (3, 3)
        vbse_img = vbse_gen.get_images_from_grid()
        with pytest.raises(
            ValueError, match="`lazy_output=True` requires `inplace=False`"
        ):
            _ = vbse_img.normalize_intensity(lazy_output=True)

        vbse_img2 = vbse_img.normalize_intensity(inplace=False, lazy_output=True)
        assert isinstance(vbse_img2, kp.signals.LazyVirtualBSEImage)

        vbse_img3 = vbse_img.as_lazy()
        vbse_img4 = vbse_img3.normalize_intensity(inplace=False, lazy_output=False)
        assert isinstance(vbse_img4, kp.signals.VirtualBSEImage)

    def test_adaptive_histogram_equalization(self):
        data = np.random.random(4800).reshape((40, 30, 2, 2))
        s = kp.signals.EBSD(data)
        vbse_img = s.get_virtual_bse_intensity(hs.roi.RectangularROI(0, 0, 2, 2))
        vbse_img.rescale_intensity(dtype_out=np.uint8)
        vbse_img.adaptive_histogram_equalization()
        assert abs(np.unique(vbse_img.data).size - 255) <= 10

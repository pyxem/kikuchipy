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

import numpy as np
import pytest

import kikuchipy.filters.fft_barnes as barnes
from kikuchipy.filters.window import Window


class TestBarnesFFTFilter:
    @pytest.mark.parametrize(
        "shape, expected_offset", [((10, 10), (5, 5)), ((17, 31), (8, 15))],
    )
    def test_offset_before_fft(self, shape, expected_offset):
        offset = barnes._offset_before_fft(shape)

        assert offset == expected_offset
        assert isinstance(offset, tuple)
        assert len(offset) == 2
        assert np.issubdtype(type(offset[0]), np.signedinteger)

    @pytest.mark.parametrize(
        "shape, expected_offset", [((10, 10), (4, 4)), ((17, 31), (8, 15))],
    )
    def test_offset_after_ifft(self, shape, expected_offset):
        offset = barnes._offset_after_ifft(shape)

        assert offset == expected_offset
        assert isinstance(offset, tuple)
        assert len(offset) == 2
        assert np.issubdtype(type(offset[0]), np.signedinteger)

    @pytest.mark.parametrize(
        "image_shape, expected_shape",
        [
            ((60, 60), (80, 90)),
            ((96, 96), (120, 120)),
            ((800, 600), (864, 625)),
        ],
    )
    def test_pad_image(self, image_shape, expected_shape):
        p = np.ones(image_shape, dtype=np.uint8)
        w = Window("gaussian", shape=(21, 23), std=5)

        fft_shape, window_rfft, off1, off2 = barnes._fft_filter_setup(
            image_shape=p.shape, window=w,
        )
        p_padded = barnes._pad_image(
            image=p,
            fft_shape=fft_shape,
            window_shape=w.shape,
            offset_before_fft=off1,
        )

        sy, sx = p.shape
        assert p_padded.shape == expected_shape
        assert np.allclose(np.sum(p_padded[:sy, :sx]), np.sum(p))

    def test_pad_window(self):
        window_shape = (5, 5)
        ky, kx = window_shape
        w = Window("gaussian", shape=window_shape)
        fft_shape = (10, 10)
        w_padded = barnes._pad_window(window=w, fft_shape=fft_shape)

        assert w_padded.shape == fft_shape
        assert np.allclose(np.sum(w_padded[:ky, :kx]), np.sum(w), atol=1e-5)

    def test_pad_window_raises(self):
        w = Window("gaussian", shape=(5, 5))
        with pytest.raises(ValueError, match="could not broadcast input array"):
            _ = barnes._pad_window(window=w, fft_shape=(4, 4))

    def test_fft_filter_setup(self):
        p = np.ones((60, 60), dtype=np.uint8)
        w = Window("gaussian", shape=(10, 10), std=5)

        fft_shape, window_rfft, off1, off2 = barnes._fft_filter_setup(
            image_shape=p.shape, window=w,
        )

        assert isinstance(fft_shape, tuple)
        assert fft_shape == (72, 72)

        assert np.sum(window_rfft.imag) != 0

        assert isinstance(off1, tuple)
        assert len(off1) == 2
        assert np.issubdtype(type(off1[0]), np.signedinteger)

        assert isinstance(off2, tuple)
        assert len(off2) == 2
        assert np.issubdtype(type(off2[0]), np.signedinteger)

    def test_fft_filter_private(self):
        p = np.ones((60, 60), dtype=np.uint8)
        w = Window("gaussian", shape=(10, 10), std=5)

        fft_shape, window_rfft, off1, off2 = barnes._fft_filter_setup(
            image_shape=p.shape, window=w,
        )

        p_filtered = barnes._fft_filter(
            image=p,
            fft_shape=fft_shape,
            window_shape=w.shape,
            transfer_function=window_rfft,
            offset_before_fft=off1,
            offset_after_ifft=off2,
        )

        assert p_filtered.shape == p.shape

    @pytest.mark.parametrize(
        "image, w, answer",
        [
            (
                np.array([[1, 2], [3, 4]]),
                np.array([[1]]),
                np.array([[1, 2], [3, 4]]),
            ),
            (
                np.array([[1, 2], [3, 4]]),
                np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
                np.array([[8, 11], [14, 17]]),
            ),
            (
                np.array([[1, 2], [3, 4]]),
                np.array([[1, 1, 1, 1, 1]]),
                np.array([[7, 8], [17, 18]]),
            ),
            (
                np.array([[1, 2], [3, 4]]),
                np.array([[1, 1, 1, 1, 1]] * 5),
                np.array([[55, 60], [65, 70]]),
            ),
            (
                np.array([[1]]),
                np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
                np.array([[5]]),
            ),
            (np.array([[2]]), np.array([[3]]), np.array([[6]])),
            (
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
                np.array([[9, 13, 17], [21, 25, 29], [33, 37, 41]]),
            ),
            (
                np.array([[1, 2], [3, 4]]),
                np.array([[1, 2], [3, 4]]),
                np.array([[10, 16], [24, 30]]),
            ),
        ],
    )
    def test_fft_filter(self, image, w, answer):
        assert np.allclose(barnes.fft_filter(image, w), answer)

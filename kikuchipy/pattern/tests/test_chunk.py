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
from scipy.ndimage import gaussian_filter

from kikuchipy.filters.fft_barnes import _fft_filter
from kikuchipy.filters.window import Window
from kikuchipy.pattern._pattern import (
    _dynamic_background_frequency_space_setup,
    fft_filter,
    fft_spectrum,
)
from kikuchipy.pattern import chunk
from kikuchipy.signals.util._dask import get_dask_array


class TestGetDynamicBackgroundChunk:
    @pytest.mark.parametrize(
        "std, answer",
        [
            (1, np.array([[5, 5, 5], [5, 4, 3], [4, 3, 2]], dtype=np.uint8)),
            (2, np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]], dtype=np.uint8)),
        ],
    )
    def test_get_dynamic_background_spatial(self, dummy_signal, std, answer):
        filter_func = gaussian_filter
        kwargs = {"sigma": std}

        dtype_out = np.uint8
        dask_array = get_dask_array(dummy_signal, dtype=np.float32)

        background = dask_array.map_blocks(
            func=chunk.get_dynamic_background,
            filter_func=filter_func,
            dtype_out=dtype_out,
            dtype=dtype_out,
            **kwargs,
        )
        assert background.dtype == dtype_out
        assert np.allclose(background[0, 0].compute(), answer)

    @pytest.mark.parametrize(
        "std, answer",
        [
            (1, np.array([[5, 5, 5], [5, 5, 4], [5, 4, 3]], dtype=np.uint8)),
            (2, np.array([[5, 5, 4], [5, 4, 4], [5, 4, 3]], dtype=np.uint8)),
            (
                1,
                np.array(
                    [
                        [5.3672, 5.4999, 5.4016],
                        [5.7932, 5.4621, 4.8999],
                        [5.8638, 4.7310, 3.3672],
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
    )
    def test_get_dynamic_background_frequency(self, dummy_signal, std, answer):
        dtype_out = answer.dtype

        dask_array = get_dask_array(dummy_signal, dtype=np.float32)

        kwargs = {}
        (
            kwargs["fft_shape"],
            kwargs["window_shape"],
            kwargs["transfer_function"],
            kwargs["offset_before_fft"],
            kwargs["offset_after_ifft"],
        ) = _dynamic_background_frequency_space_setup(
            pattern_shape=dummy_signal._signal_shape_rc,
            std=std,
            truncate=4.0,
        )

        background = dask_array.map_blocks(
            func=chunk.get_dynamic_background,
            filter_func=_fft_filter,
            dtype_out=dtype_out,
            dtype=dtype_out,
            **kwargs,
        )

        # Check for correct data type and gives expected output intensities
        assert background.dtype == dtype_out
        assert np.allclose(background[0, 0].compute(), answer, atol=1e-4)

    @pytest.mark.parametrize(
        "answer",
        [
            np.array(
                [
                    [5.2676, 5.0783, 4.8443],
                    [5.2065, 4.8083, 4.3654],
                    [5.0473, 4.4041, 3.7162],
                ],
                dtype=np.float32,
            ),
            np.array([[5, 5, 4], [5, 4, 4], [5, 4, 3]], dtype=np.uint16),
        ],
    )
    def test_get_dynamic_background_dtype_out(self, dummy_signal, answer):
        dtype_out = answer.dtype
        dummy_signal.data = dummy_signal.data.astype(dtype_out)

        kwargs = {}
        (
            kwargs["fft_shape"],
            kwargs["window_shape"],
            kwargs["transfer_function"],
            kwargs["offset_before_fft"],
            kwargs["offset_after_ifft"],
        ) = _dynamic_background_frequency_space_setup(
            pattern_shape=dummy_signal._signal_shape_rc,
            std=2,
            truncate=4.0,
        )

        background = chunk.get_dynamic_background(
            patterns=dummy_signal.data, filter_func=_fft_filter, **kwargs
        )

        assert isinstance(background, np.ndarray)
        assert background.dtype == dtype_out
        assert np.allclose(background[0, 0], answer, atol=1e-4)


class TestFFTFilterChunk:
    @pytest.mark.parametrize(
        "shift, transfer_function, kwargs, dtype_out, expected_spectrum_sum",
        [
            (True, "modified_hann", {}, None, 8924.0208),
            (
                True,
                "lowpass",
                {"cutoff": 30, "cutoff_width": 15},
                np.float64,
                8231.9982,
            ),
            (
                False,
                "highpass",
                {"cutoff": 2, "cutoff_width": 1},
                np.float32,
                10188.1395,
            ),
            (False, "gaussian", {"sigma": 2}, None, 414.7418),
        ],
    )
    def test_fft_filter(
        self, shift, transfer_function, kwargs, dtype_out, expected_spectrum_sum
    ):
        dtype_in = np.float64

        shape = (101, 101)
        p = np.ones((4,) + shape, dtype=dtype_in)
        this_id = 2
        p[this_id, 50, 50] = 2

        w = Window(transfer_function, shape=shape, **kwargs)

        filter_func = fft_filter

        p_fft = chunk.fft_filter(
            patterns=p,
            filter_func=filter_func,
            transfer_function=w,
            shift=shift,
            dtype_out=dtype_out,
        )

        this_fft = p_fft[this_id]

        if dtype_out is None:
            dtype_out = np.float64

        assert this_fft.dtype == dtype_out
        assert np.allclose(
            np.sum(fft_spectrum.py_func(this_fft)), expected_spectrum_sum, atol=1e-4
        )

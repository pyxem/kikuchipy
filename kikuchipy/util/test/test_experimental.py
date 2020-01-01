# -*- coding: utf-8 -*-
# Copyright 2019-2020 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import os

import numpy as np
import pytest

import kikuchipy as kp

# Expected output intensities from various pattern processing methods
RESCALED_UINT8 = np.array(
    [[182, 218, 182], [255, 218, 182], [218, 36, 0]], dtype=np.uint8
)
RESCALED_FLOAT32 = np.array(
    [
        [0.714286, 0.857143, 0.714286],
        [1.0, 0.857143, 0.714286],
        [0.857143, 0.142857, 0.0],
    ],
    dtype=np.float32,
)
RESCALED_UINT8_0100 = np.array(
    [[71, 85, 71], [100, 85, 71], [85, 14, 0]], dtype=np.uint8
)
STATIC_CORR_UINT8 = np.array(
    [[0, 2, 0], [3, 3, 1], [2, 255, 255]], dtype=np.uint8
)
DYNAMIC_CORR_UINT8 = np.array(
    [[0, 1, 1], [2, 1, 0], [1, 255, 253]], dtype=np.uint8
)
ADAPT_EQ_UINT8 = np.array(
    [[127, 223, 127], [255, 223, 31], [223, 31, 0]], dtype=np.uint8
)


class TestExperimental:
    @pytest.mark.parametrize(
        "dtype_out, out_range, answer",
        [
            (np.uint8, None, RESCALED_UINT8),
            (np.float32, None, RESCALED_FLOAT32),
            (None, None, RESCALED_UINT8),
            (np.complex, None, RESCALED_UINT8),
            (np.uint8, (0, 100), RESCALED_UINT8_0100),
        ],
    )
    def test_rescale_pattern(self, dummy_signal, dtype_out, out_range, answer):
        pattern = dummy_signal.inav[0, 0]
        pattern_dask = kp.util.dask._get_dask_array(pattern)

        # Check for accepted data types
        if dtype_out == np.complex:
            with pytest.raises(KeyError, match="Could not set output"):
                kp.util.experimental._rescale_pattern(
                    pattern_dask,
                    in_range=None,
                    out_range=out_range,
                    dtype_out=dtype_out,
                )
            return 0  # So that the test ends here
        else:
            rescaled_pattern = kp.util.experimental._rescale_pattern(
                pattern_dask,
                in_range=None,
                out_range=out_range,
                dtype_out=dtype_out,
            )

        # Check for correct data type and gives expected output intensities
        if dtype_out is not None:
            assert rescaled_pattern.dtype == dtype_out
        np.testing.assert_almost_equal(
            rescaled_pattern.compute(), answer, decimal=6
        )

    def test_static_background_correction_chunk(
        self, dummy_signal, dummy_background
    ):
        dask_array = kp.util.dask._get_dask_array(dummy_signal)
        dtype_out = dask_array.dtype
        corrected_patterns = dask_array.map_blocks(
            kp.util.experimental._static_background_correction_chunk,
            static_bg=dummy_background,
            operation="subtract",
            dtype=dtype_out,
        )

        # Check for correct data type and gives expected output intensities
        assert corrected_patterns.dtype == dtype_out
        np.testing.assert_almost_equal(
            corrected_patterns[0, 0].compute(), STATIC_CORR_UINT8
        )

    def test_dynamic_background_correction_chunk(self, dummy_signal):
        dask_array = kp.util.dask._get_dask_array(dummy_signal)
        dtype_out = dask_array.dtype
        corrected_patterns = dask_array.map_blocks(
            kp.util.experimental._dynamic_background_correction_chunk,
            sigma=2,
            operation="subtract",
            dtype=dtype_out,
        )

        # Check for correct data type and gives expected output intensities
        assert corrected_patterns.dtype == dtype_out
        np.testing.assert_almost_equal(
            corrected_patterns[0, 0].compute(), DYNAMIC_CORR_UINT8
        )

    def test_adaptive_histogram_equalization_chunk(self, dummy_signal):
        dask_array = kp.util.dask._get_dask_array(dummy_signal)
        dtype_out = dask_array.dtype
        kernel_size = (10, 10)
        nbins = 128
        equalized_patterns = dask_array.map_blocks(
            kp.util.experimental._adaptive_histogram_equalization_chunk,
            kernel_size=kernel_size,
            nbins=nbins,
        )

        # Check for correct data type and gives expected output intensities
        assert equalized_patterns.dtype == dtype_out
        np.testing.assert_almost_equal(
            equalized_patterns[0, 0].compute(), ADAPT_EQ_UINT8
        )

    @pytest.mark.parametrize(
        "pattern_idx, template_idx, answer",
        [((0, 0), (0, 1), 0.4935737), ((0, 0), (0, 0), 1.0000000)],
    )
    def test_normalised_correlation_coefficient(
        self, dummy_signal, pattern_idx, template_idx, answer
    ):
        coefficient = kp.util.experimental.normalised_correlation_coefficient(
            pattern=dummy_signal.inav[pattern_idx].data,
            template=dummy_signal.inav[template_idx].data,
            zero_normalised=True,
        )
        np.testing.assert_almost_equal(coefficient, answer, decimal=7)

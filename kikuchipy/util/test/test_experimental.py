# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
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

import numpy as np
import pytest

import kikuchipy as kp

RESCALED_UINT8 = np.array(
    [[182, 218, 182], [255, 218, 182], [218, 36, 0]], dtype=np.uint8)
RESCALED_FLOAT32 = np.array(
    [[0.714286, 0.857143, 0.714286], [1., 0.857143, 0.714286],
     [0.857143, 0.142857, 0.]], dtype=np.float32)
RESCALED_UINT8_0100 = np.array(
    [[71, 85, 71], [100, 85, 71], [85, 14, 0]], dtype=np.uint8)
STATIC_CORR_UINT8 = np.array(
    [[0, 2, 0],[3, 3, 1], [2, 255, 255]], dtype=np.uint8)
DYNAMIC_CORR_UINT8 = np.array(
    [[0, 1, 1], [2, 1, 0], [1, 255, 253]], dtype=np.uint8)


class TestExperimental:

    @pytest.mark.parametrize(
        'dtype_out, out_range, answer',
        [(np.uint8, None, RESCALED_UINT8), (np.float32, None, RESCALED_FLOAT32),
         (None, None, RESCALED_UINT8), (np.complex, None, RESCALED_UINT8),
         (np.uint8, (0, 100), RESCALED_UINT8_0100)])
    def test_rescale_pattern(
            self, dummy_signal, dtype_out, out_range, answer):
        pattern = dummy_signal.inav[0, 0]
        pattern_dask = kp.util.dask._get_dask_array(pattern)
        if dtype_out == np.complex:
            with pytest.raises(KeyError, match='Could not set output'):
                kp.util.experimental._rescale_pattern(
                    pattern_dask, in_range=None, out_range=out_range,
                    dtype_out=dtype_out)
            return 0
        else:
            rescaled_pattern = kp.util.experimental._rescale_pattern(
                pattern_dask, in_range=None, out_range=out_range,
                dtype_out=dtype_out)
        if dtype_out != None:
            assert rescaled_pattern.dtype == dtype_out
        np.testing.assert_almost_equal(
            rescaled_pattern.compute(), answer, decimal=6)

    def test_static_background_correction_chunk(
            self, dummy_signal, dummy_background):
        dask_array = kp.util.dask._get_dask_array(dummy_signal)
        dtype_out = dask_array.dtype
        corrected_patterns = dask_array.map_blocks(
            kp.util.experimental._static_background_correction_chunk,
            static_bg=dummy_background, operation='subtract', dtype=dtype_out)

        assert corrected_patterns.dtype == dtype_out
        np.testing.assert_almost_equal(
            corrected_patterns[0, 0].compute(), STATIC_CORR_UINT8)

    def test_dynamic_background_correction_chunk(self, dummy_signal):
        dask_array = kp.util.dask._get_dask_array(dummy_signal)
        dtype_out = dask_array.dtype
        corrected_patterns = dask_array.map_blocks(
            kp.util.experimental._dynamic_background_correction_chunk,
            sigma=2, operation='subtract', dtype=dtype_out)

        assert corrected_patterns.dtype == dtype_out
        np.testing.assert_almost_equal(
            corrected_patterns[0, 0].compute(), DYNAMIC_CORR_UINT8)

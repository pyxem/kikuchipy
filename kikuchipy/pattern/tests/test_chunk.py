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

import dask.array as da
import numpy as np
import pytest
from scipy.ndimage import convolve, gaussian_filter
from skimage import __version__ as skimage_version
from skimage.util.dtype import dtype_range

from kikuchipy.filters.fft_barnes import _fft_filter
from kikuchipy.filters.window import Window
from kikuchipy.pattern._pattern import (
    _dynamic_background_frequency_space_setup,
    fft_filter,
    fft_spectrum,
)
from kikuchipy.pattern import chunk
from kikuchipy.signals.util._dask import _get_dask_array


# Expected output intensities from various image processing methods
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
STATIC_SUB_UINT8 = np.array(
    [[127, 212, 127], [255, 255, 170], [212, 0, 0]], dtype=np.uint8
)
STATIC_SUB_SCALEBG_UINT8 = np.array(
    [[15, 150, 15], [180, 255, 120], [150, 0, 75]], dtype=np.uint8
)
STATIC_DIV_UINT8 = np.array(
    [[127, 191, 127], [223, 255, 159], [191, 31, 0]], dtype=np.uint8
)
DYN_CORR_UINT8_SPATIAL_STD2 = np.array(
    [[170, 215, 181], [255, 221, 188], [221, 32, 0]], dtype=np.uint8
)
DYN_CORR_UINT8_SPATIAL_STD1 = np.array(
    [[120, 197, 156], [255, 241, 223], [226, 0, 9]], dtype=np.uint8
)
DYN_CORR_UINT8_FREQUENCY_STD1_TRUNCATE3 = np.array(
    [[111, 191, 141], [255, 253, 243], [221, 0, 38]], dtype=np.uint8
)
DYN_CORR_UINT8_FREQUENCY_STD2_TRUNCATE4 = np.array(
    [[159, 214, 179], [255, 227, 201], [216, 14, 0]], dtype=np.uint8
)
DYN_CORR_UINT16_SPATIAL_STD2 = np.array(
    [[43928, 55293, 46544], [65535, 56974, 48412], [56975, 8374, 0]],
    dtype=np.uint16,
)
DYN_CORR_UINT8_SPATIAL_STD2_OMAX250 = np.array(
    [[167, 210, 177], [250, 217, 184], [217, 31, 0]], dtype=np.uint8,
)
ADAPT_EQ_UINT8_SKIMAGE16 = np.array(
    [[127, 223, 127], [255, 223, 31], [223, 31, 0]], dtype=np.uint8
)
ADAPT_EQ_UINT8_SKIMAGE17 = np.array(
    [[92, 215, 92], [255, 215, 92], [215, 26, 0]], dtype=np.uint8
)
ADAPT_EQ_UINT8 = ADAPT_EQ_UINT8_SKIMAGE16
if skimage_version[2:4] == str(17):
    ADAPT_EQ_UINT8 = ADAPT_EQ_UINT8_SKIMAGE17


class TestRescaleIntensityChunk:
    @pytest.mark.parametrize(
        "dtype_out, answer",
        [
            (
                np.uint8,
                np.array([[182, 218, 182], [255, 218, 182], [218, 36, 0]]),
            ),
            (
                np.float32,
                np.array(
                    [
                        [0.4285, 0.7142, 0.4285],
                        [1, 0.7142, 0.4285],
                        [0.7142, -0.7142, -1],
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
    )
    def test_rescale_intensity(self, dummy_signal, dtype_out, answer):
        dask_array = _get_dask_array(dummy_signal, dtype=np.float32)

        rescaled_patterns = dask_array.map_blocks(
            func=chunk.rescale_intensity, dtype_out=dtype_out, dtype=dtype_out,
        )

        assert isinstance(rescaled_patterns, da.Array)
        assert rescaled_patterns.dtype == dtype_out
        assert np.allclose(rescaled_patterns[0, 0].compute(), answer, atol=1e-4)

    @pytest.mark.parametrize(
        "out_range, dtype_out, answer",
        [
            (
                (0, 255),
                np.uint8,
                np.array([[182, 218, 182], [255, 218, 182], [218, 36, 0]]),
            ),
            (
                (5, 200),
                np.uint8,
                np.array([[144, 172, 144], [200, 172, 144], [172, 32, 5]]),
            ),
            (
                (-1, 1),
                np.float32,
                np.array(
                    [
                        [0.4285, 0.7142, 0.4285],
                        [1.0, 0.7142, 0.4285],
                        [0.7142, -0.7142, -1],
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
    )
    def test_rescale_intensity_out_range(
        self, dummy_signal, out_range, dtype_out, answer
    ):
        dummy_signal.data = dummy_signal.data.astype(np.float32)

        rescaled_patterns = chunk.rescale_intensity(
            patterns=dummy_signal.data,
            out_range=out_range,
            dtype_out=dtype_out,
        )

        assert isinstance(rescaled_patterns, np.ndarray)
        assert rescaled_patterns.dtype == dtype_out
        assert np.allclose(rescaled_patterns[0, 0], answer, atol=1e-4)

    @pytest.mark.parametrize(
        "in_range, answer",
        [
            ((2, 250), np.array([[3, 4, 3], [5, 4, 3], [4, 0, 0]])),
            ((3, 250), np.array([[2, 3, 2], [4, 3, 2], [3, 0, 0]])),
        ],
    )
    def test_rescale_intensity_in_range(self, dummy_signal, in_range, answer):
        dtype_out = dummy_signal.data.dtype
        dask_array = _get_dask_array(dummy_signal, dtype=np.float32)

        rescaled_patterns = dask_array.map_blocks(
            func=chunk.rescale_intensity,
            in_range=in_range,
            dtype_out=dtype_out,
            dtype=dtype_out,
        )

        assert isinstance(rescaled_patterns, da.Array)
        assert rescaled_patterns.dtype == dtype_out
        assert np.allclose(rescaled_patterns[0, 0].compute(), answer)

    @pytest.mark.parametrize(
        "percentiles, answer",
        [
            (
                (10, 90),
                np.array([[198, 245, 198], [254, 245, 198], [245, 9, 0]]),
            ),
            (
                (1, 99),
                np.array([[183, 220, 183], [255, 220, 183], [220, 34, 0]]),
            ),
        ],
    )
    def test_rescale_intensity_percentiles(
        self, dummy_signal, percentiles, answer
    ):
        dtype_out = dummy_signal.data.dtype
        dask_array = _get_dask_array(dummy_signal, dtype=np.float32)

        rescaled_patterns = dask_array.map_blocks(
            func=chunk.rescale_intensity,
            percentiles=percentiles,
            dtype_out=dtype_out,
            dtype=dtype_out,
        )

        p1 = rescaled_patterns[0, 0].compute()
        p2 = rescaled_patterns[0, 1].compute()

        assert isinstance(rescaled_patterns, da.Array)
        assert rescaled_patterns.dtype == dtype_out
        assert np.allclose(p1, answer)
        assert not np.allclose(p1, p2, atol=1)


class TestRemoveStaticBackgroundChunk:
    @pytest.mark.parametrize(
        "operation_func, answer",
        [(np.subtract, STATIC_SUB_UINT8), (np.divide, STATIC_DIV_UINT8),],
    )
    def test_remove_static_background_chunk(
        self, dummy_signal, dummy_background, operation_func, answer
    ):
        dtype_out = dummy_signal.data.dtype.type
        dtype_proc = np.float32

        dask_array = _get_dask_array(dummy_signal, dtype=dtype_proc)
        corrected_patterns = dask_array.map_blocks(
            func=chunk.remove_static_background,
            static_bg=dummy_background.astype(dtype_proc),
            operation_func=operation_func,
            dtype_out=dtype_out,
            dtype=dtype_out,
        )

        # Check for correct data type and gives expected output intensities
        assert corrected_patterns.dtype == dtype_out
        assert isinstance(corrected_patterns, da.Array)
        assert np.allclose(corrected_patterns[0, 0].compute(), answer)

    def test_remove_static_background_chunk_out_range(
        self, dummy_signal, dummy_background
    ):
        dtype_out = dummy_signal.data.dtype.type
        dtype_proc = np.float32

        out_range = (0, dtype_range[dtype_out][-1])

        dask_array = _get_dask_array(dummy_signal, dtype=dtype_proc)

        corrected_patterns = dask_array.map_blocks(
            func=chunk.remove_static_background,
            static_bg=dummy_background.astype(dtype_proc),
            operation_func=np.subtract,
            dtype_out=dtype_out,
            out_range=out_range,
            dtype=dtype_out,
        )

        assert corrected_patterns.dtype == dtype_out
        assert np.allclose(corrected_patterns[0, 0].compute(), STATIC_SUB_UINT8)

    def test_remove_static_background_chunk_scalebg(
        self, dummy_signal, dummy_background
    ):
        dtype_out = dummy_signal.data.dtype.type
        dtype_proc = np.float32

        dask_array = _get_dask_array(dummy_signal, dtype=dtype_proc)

        corrected_patterns = dask_array.map_blocks(
            func=chunk.remove_static_background,
            static_bg=dummy_background.astype(dtype_proc),
            operation_func=np.subtract,
            dtype_out=dtype_out,
            scale_bg=True,
            dtype=dtype_out,
        )

        assert corrected_patterns.dtype == dtype_out
        assert np.allclose(
            corrected_patterns[0, 0].compute(), STATIC_SUB_SCALEBG_UINT8
        )

    @pytest.mark.parametrize(
        "dtype_out, answer",
        [
            (
                np.float32,
                np.array([[0, 0.6666, 0], [1, 1, 0.3333], [0.6666, -1, -1]]),
            ),
            (np.uint16, np.array([[0, 2, 0], [3, 3, 1], [2, 65535, 65535]])),
        ],
    )
    def test_remove_static_background_chunk_dtype_out(
        self, dummy_signal, dummy_background, dtype_out, answer
    ):
        dummy_signal.data = dummy_signal.data.astype(dtype_out)
        dummy_background = dummy_background.astype(dtype_out)

        corrected_patterns = chunk.remove_static_background(
            patterns=dummy_signal.data,
            static_bg=dummy_background,
            operation_func=np.subtract,
        )

        assert corrected_patterns.dtype == dtype_out
        assert np.allclose(corrected_patterns[0, 0], answer, atol=1e-4)


class TestRemoveDynamicBackgroundChunk:
    @pytest.mark.parametrize(
        "std, answer",
        [(1, DYN_CORR_UINT8_SPATIAL_STD1), (2, DYN_CORR_UINT8_SPATIAL_STD2)],
    )
    def test_remove_dynamic_background_spatial(self, dummy_signal, std, answer):
        dtype_out = dummy_signal.data.dtype.type

        dask_array = _get_dask_array(dummy_signal, dtype=np.float32)

        kwargs = {"sigma": std}

        corrected_patterns = dask_array.map_blocks(
            func=chunk.remove_dynamic_background,
            filter_func=gaussian_filter,
            operation_func=np.subtract,
            dtype_out=dtype_out,
            dtype=dtype_out,
            **kwargs,
        )

        # Check for correct data type and gives expected output intensities
        assert corrected_patterns.dtype == dtype_out
        assert np.allclose(corrected_patterns[0, 0].compute(), answer)

    def test_remove_dynamic_background_spatial_uint16(self, dummy_signal):
        dtype_out = np.uint16

        dask_array = _get_dask_array(dummy_signal, dtype=np.float32)

        corrected_patterns = dask_array.map_blocks(
            func=chunk.remove_dynamic_background,
            filter_func=gaussian_filter,
            operation_func=np.subtract,
            dtype_out=dtype_out,
            dtype=dtype_out,
            sigma=2,
        )

        # Check for correct data type and gives expected output intensities
        assert corrected_patterns.dtype == dtype_out
        assert np.allclose(
            corrected_patterns[0, 0].compute(), DYN_CORR_UINT16_SPATIAL_STD2
        )

    @pytest.mark.parametrize(
        "std, truncate, answer",
        [
            (1, 3, DYN_CORR_UINT8_FREQUENCY_STD1_TRUNCATE3),
            (2, 4, DYN_CORR_UINT8_FREQUENCY_STD2_TRUNCATE4),
        ],
    )
    def test_remove_dynamic_background_frequency(
        self, dummy_signal, std, truncate, answer
    ):
        dtype_out = dummy_signal.data.dtype.type

        dask_array = _get_dask_array(dummy_signal, dtype=np.float32)

        kwargs = {}
        (
            kwargs["fft_shape"],
            kwargs["window_shape"],
            kwargs["transfer_function"],
            kwargs["offset_before_fft"],
            kwargs["offset_after_ifft"],
        ) = _dynamic_background_frequency_space_setup(
            pattern_shape=dummy_signal.axes_manager.signal_shape[::-1],
            std=std,
            truncate=truncate,
        )

        corrected_patterns = dask_array.map_blocks(
            func=chunk.remove_dynamic_background,
            filter_func=_fft_filter,
            operation_func=np.subtract,
            dtype_out=dtype_out,
            dtype=dtype_out,
            **kwargs,
        )

        # Check for correct data type and gives expected output intensities
        assert corrected_patterns.dtype == dtype_out
        assert np.allclose(corrected_patterns[0, 0].compute(), answer)

    @pytest.mark.parametrize(
        "omax, answer",
        [
            (255, DYN_CORR_UINT8_SPATIAL_STD2),
            (250, DYN_CORR_UINT8_SPATIAL_STD2_OMAX250),
        ],
    )
    def test_remove_dynamic_background_out_range(
        self, dummy_signal, omax, answer
    ):
        dtype_out = dummy_signal.data.dtype.type

        out_range = (0, omax)

        dask_array = _get_dask_array(dummy_signal, dtype=np.float32)

        corrected_patterns = dask_array.map_blocks(
            func=chunk.remove_dynamic_background,
            filter_func=gaussian_filter,
            operation_func=np.subtract,
            sigma=2,
            out_range=out_range,
            dtype_out=dtype_out,
            dtype=dtype_out,
        )

        assert corrected_patterns.dtype == dtype_out
        assert corrected_patterns.max().compute() == omax
        assert np.allclose(corrected_patterns[0, 0].compute(), answer)

    @pytest.mark.parametrize(
        "answer",
        [
            np.array(
                [
                    [0.3405, 0.6874, 0.4204],
                    [1, 0.7387, 0.4774],
                    [0.7387, -0.7444, -1],
                ],
                dtype=np.float32,
            ),
            np.array(
                [[0, 1, 1], [2, 1, 0], [1, 65535, 65533]], dtype=np.uint16,
            ),
        ],
    )
    def test_remove_dynamic_background_dtype_out(self, dummy_signal, answer):
        dtype_out = answer.dtype
        dummy_signal.data = dummy_signal.data.astype(dtype_out)

        corrected_patterns = chunk.remove_dynamic_background(
            patterns=dummy_signal.data,
            filter_func=gaussian_filter,
            operation_func=np.subtract,
            sigma=2,
        )

        assert corrected_patterns.dtype == dtype_out
        assert np.allclose(corrected_patterns[0, 0], answer, atol=1e-4)


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
        dask_array = _get_dask_array(dummy_signal, dtype=np.float32)

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

        dask_array = _get_dask_array(dummy_signal, dtype=np.float32)

        kwargs = {}
        (
            kwargs["fft_shape"],
            kwargs["window_shape"],
            kwargs["transfer_function"],
            kwargs["offset_before_fft"],
            kwargs["offset_after_ifft"],
        ) = _dynamic_background_frequency_space_setup(
            pattern_shape=dummy_signal.axes_manager.signal_shape[::-1],
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
            pattern_shape=dummy_signal.axes_manager.signal_shape[::-1],
            std=2,
            truncate=4.0,
        )

        background = chunk.get_dynamic_background(
            patterns=dummy_signal.data, filter_func=_fft_filter, **kwargs,
        )

        assert isinstance(background, np.ndarray)
        assert background.dtype == dtype_out
        assert np.allclose(background[0, 0], answer, atol=1e-4)


class TestAdaptiveHistogramEqualizationChunk:
    def test_adaptive_histogram_equalization_chunk(self, dummy_signal):
        dask_array = _get_dask_array(dummy_signal)
        dtype_out = dask_array.dtype
        kernel_size = (10, 10)
        nbins = 128
        equalized_patterns = dask_array.map_blocks(
            func=chunk.adaptive_histogram_equalization,
            kernel_size=kernel_size,
            nbins=nbins,
        )

        # Check for correct data type and gives expected output intensities
        assert equalized_patterns.dtype == dtype_out
        assert np.allclose(equalized_patterns[0, 0].compute(), ADAPT_EQ_UINT8)


class TestAverageNeighbourPatternsChunk:
    @pytest.mark.parametrize("dtype_in", [None, np.uint8])
    def test_average_neighbour_patterns_chunk(self, dummy_signal, dtype_in):
        w = Window()

        # Get array to operate on
        dask_array = _get_dask_array(dummy_signal)
        dtype_out = dask_array.dtype

        # Get sum of window data for each image
        nav_shape = dummy_signal.axes_manager.navigation_shape
        w_sums = convolve(
            input=np.ones(nav_shape[::-1]),
            weights=w.data,
            mode="constant",
            cval=0,
        )

        for i in range(dummy_signal.axes_manager.signal_dimension):
            w_sums = np.expand_dims(w_sums, axis=w_sums.ndim)
        w_sums = da.from_array(w_sums, chunks=dask_array.chunksize)

        # Add signal dimensions to window array to enable its use with Dask's
        # map_blocks()
        w = w.reshape(
            w.shape + (1,) * dummy_signal.axes_manager.signal_dimension
        )

        averaged_patterns = dask_array.map_blocks(
            func=chunk.average_neighbour_patterns,
            window_sums=w_sums,
            window=w,
            dtype_out=dtype_in,
            dtype=dtype_out,
        )

        answer = np.array([7, 4, 6, 6, 3, 7, 7, 3, 2], dtype=np.uint8).reshape(
            (3, 3)
        )

        # Check for correct data type and gives expected output intensities
        assert averaged_patterns.dtype == dtype_out
        assert np.allclose(averaged_patterns[0, 0].compute(), answer)


class TestGetImageQualityChunk:
    @pytest.mark.parametrize(
        "normalize, answer",
        [
            (
                True,
                np.array(
                    [
                        [-0.0241, -0.0625, -0.0052],
                        [-0.0317, -0.0458, -0.0956],
                        [-0.1253, 0.0120, -0.2385],
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                False,
                np.array(
                    [
                        [0.2694, 0.2926, 0.2299],
                        [0.2673, 0.1283, 0.2032],
                        [0.1105, 0.2671, 0.2159],
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_get_image_quality_chunk(self, dummy_signal, normalize, answer):
        iq = chunk.get_image_quality(
            patterns=dummy_signal.data, normalize=normalize,
        )

        assert np.allclose(iq, answer, atol=1e-4)

    def test_get_image_quality_chunk_white_noise(self):
        p = np.random.random((4, 1001, 1001))
        iq = chunk.get_image_quality(patterns=p, normalize=True)
        assert np.allclose(iq, 0, atol=1e-2)

    def test_get_image_quality_flat(self):
        p = np.ones((4, 1001, 1001))
        iq = chunk.get_image_quality(patterns=p, normalize=False)
        assert np.allclose(iq, 1, atol=1e-2)


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
            np.sum(fft_spectrum.py_func(this_fft)),
            expected_spectrum_sum,
            atol=1e-4,
        )


class TestNormalizeIntensityChunk:
    @pytest.mark.parametrize(
        "num_std, divide_by_square_root, dtype_out, answer",
        [
            (
                1,
                True,
                np.float32,
                np.array(
                    [
                        [0.0653, 0.2124, 0.0653],
                        [0.3595, 0.2124, 0.0653],
                        [0.2124, -0.5229, -0.6700],
                    ]
                ),
            ),
            (
                2,
                True,
                np.float32,
                np.array(
                    [
                        [0.0326, 0.1062, 0.0326],
                        [0.1797, 0.1062, 0.0326],
                        [0.1062, -0.2614, -0.3350],
                    ]
                ),
            ),
            (
                1,
                False,
                np.float32,
                np.array(
                    [
                        [0.1961, 0.6373, 0.1961],
                        [1.0786, 0.6373, 0.1961],
                        [0.6373, -1.5689, -2.0101],
                    ]
                ),
            ),
            (1, False, None, np.array([[0, 0, 0], [1, 0, 0], [0, -1, -2]])),
        ],
    )
    def test_normalize_intensity(
        self, dummy_signal, num_std, divide_by_square_root, dtype_out, answer
    ):
        if dtype_out is None:
            dummy_signal.data = dummy_signal.data.astype(np.int8)

        normalized_patterns = chunk.normalize_intensity(
            patterns=dummy_signal.data,
            num_std=num_std,
            divide_by_square_root=divide_by_square_root,
            dtype_out=dtype_out,
        )

        if dtype_out is None:
            dtype_out = dummy_signal.data.dtype
        else:
            assert np.allclose(np.mean(normalized_patterns), 0, atol=1e-6)

        assert normalized_patterns.dtype == dtype_out
        assert isinstance(normalized_patterns, np.ndarray)
        assert np.allclose(normalized_patterns[0, 0], answer, atol=1e-4)

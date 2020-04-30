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
from skimage.util.dtype import dtype_range

import kikuchipy as kp
from kikuchipy.util.barnes_fftfilter import _fft_filter


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
ADAPT_EQ_UINT8 = np.array(
    [[127, 223, 127], [255, 223, 31], [223, 31, 0]], dtype=np.uint8
)


class TestRescaleIntensityChunk:
    def test_rescale_intensity_chunk(self):
        pass


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

        dask_array = kp.util.dask._get_dask_array(
            dummy_signal, dtype=dtype_proc
        )
        corrected_patterns = dask_array.map_blocks(
            func=kp.util.chunk.remove_static_background,
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

        dask_array = kp.util.dask._get_dask_array(
            dummy_signal, dtype=dtype_proc
        )

        corrected_patterns = dask_array.map_blocks(
            func=kp.util.chunk.remove_static_background,
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

        dask_array = kp.util.dask._get_dask_array(
            dummy_signal, dtype=dtype_proc
        )

        corrected_patterns = dask_array.map_blocks(
            func=kp.util.chunk.remove_static_background,
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


class TestRemoveDynamicBackgroundChunk:
    @pytest.mark.parametrize(
        "std, answer",
        [(1, DYN_CORR_UINT8_SPATIAL_STD1), (2, DYN_CORR_UINT8_SPATIAL_STD2)],
    )
    def test_remove_dynamic_background_chunk_spatial(
        self, dummy_signal, std, answer
    ):
        dtype_out = dummy_signal.data.dtype.type

        dask_array = kp.util.dask._get_dask_array(
            dummy_signal, dtype=np.float32
        )

        kwargs = {"sigma": std}

        corrected_patterns = dask_array.map_blocks(
            func=kp.util.chunk.remove_dynamic_background,
            filter_func=gaussian_filter,
            operation_func=np.subtract,
            dtype_out=dtype_out,
            dtype=dtype_out,
            **kwargs,
        )

        # Check for correct data type and gives expected output intensities
        assert corrected_patterns.dtype == dtype_out
        assert np.allclose(corrected_patterns[0, 0].compute(), answer)

    def test_remove_dynamic_background_chunk_spatial_uint16(self, dummy_signal):
        dtype_out = np.uint16

        dask_array = kp.util.dask._get_dask_array(
            dummy_signal, dtype=np.float32
        )

        corrected_patterns = dask_array.map_blocks(
            func=kp.util.chunk.remove_dynamic_background,
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
    def test_remove_dynamic_background_chunk_frequency(
        self, dummy_signal, std, truncate, answer
    ):
        dtype_out = dummy_signal.data.dtype.type

        dask_array = kp.util.dask._get_dask_array(
            dummy_signal, dtype=np.float32
        )

        kwargs = {}
        (
            kwargs["fft_shape"],
            kwargs["window_shape"],
            kwargs["window_fft"],
            kwargs["offset_before_fft"],
            kwargs["offset_after_ifft"],
        ) = kp.util.pattern._dynamic_background_frequency_space_setup(
            pattern_shape=dummy_signal.axes_manager.signal_shape[::-1],
            std=std,
            truncate=truncate,
        )

        corrected_patterns = dask_array.map_blocks(
            func=kp.util.chunk.remove_dynamic_background,
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
    def test_remove_dynamic_background_chunk_out_range(
        self, dummy_signal, omax, answer
    ):
        dtype_out = dummy_signal.data.dtype.type

        out_range = (0, omax)

        dask_array = kp.util.dask._get_dask_array(
            dummy_signal, dtype=np.float32
        )

        corrected_patterns = dask_array.map_blocks(
            func=kp.util.chunk.remove_dynamic_background,
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


class TestGetDynamicBackgroundChunk:
    @pytest.mark.parametrize(
        "std, answer",
        [
            (1, np.array([[5, 5, 5], [5, 4, 3], [4, 3, 2]], dtype=np.uint8)),
            (2, np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]], dtype=np.uint8)),
        ],
    )
    def test_get_dynamic_background_chunk_spatial(
        self, dummy_signal, std, answer
    ):
        filter_func = gaussian_filter
        kwargs = {"sigma": std}

        dtype_out = np.uint8
        dask_array = kp.util.dask._get_dask_array(
            dummy_signal, dtype=np.float32
        )

        background = dask_array.map_blocks(
            func=kp.util.chunk.get_dynamic_background,
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
                # fmt: off
                np.array(
                    [
                        [5.3672, 5.4999, 5.4016],
                        [5.7932, 5.4621, 4.8999],
                        [5.8638, 4.7310, 3.3672]
                    ],
                    dtype=np.float32,
                )
                # fmt: on
            ),
        ],
    )
    def test_get_dynamic_background_chunk_frequency(
        self, dummy_signal, std, answer
    ):

        dtype_out = answer.dtype

        dask_array = kp.util.dask._get_dask_array(
            dummy_signal, dtype=np.float32
        )

        kwargs = {}
        (
            kwargs["fft_shape"],
            kwargs["window_shape"],
            kwargs["window_fft"],
            kwargs["offset_before_fft"],
            kwargs["offset_after_ifft"],
        ) = kp.util.pattern._dynamic_background_frequency_space_setup(
            pattern_shape=dummy_signal.axes_manager.signal_shape[::-1],
            std=std,
            truncate=4.0,
        )

        background = dask_array.map_blocks(
            func=kp.util.chunk.get_dynamic_background,
            filter_func=_fft_filter,
            dtype_out=dtype_out,
            dtype=dtype_out,
            **kwargs,
        )

        # Check for correct data type and gives expected output intensities
        assert background.dtype == dtype_out
        assert np.allclose(background[0, 0].compute(), answer, atol=1e-4)


class TestAdaptiveHistogramEqualizationChunk:
    def test_adaptive_histogram_equalization_chunk(self, dummy_signal):
        dask_array = kp.util.dask._get_dask_array(dummy_signal)
        dtype_out = dask_array.dtype
        kernel_size = (10, 10)
        nbins = 128
        equalized_patterns = dask_array.map_blocks(
            func=kp.util.chunk.adaptive_histogram_equalization,
            kernel_size=kernel_size,
            nbins=nbins,
        )

        # Check for correct data type and gives expected output intensities
        assert equalized_patterns.dtype == dtype_out
        assert np.allclose(equalized_patterns[0, 0].compute(), ADAPT_EQ_UINT8)


class TestAverageNeighbourPatternsChunk:
    @pytest.mark.parametrize("dtype_in", [None, np.uint8])
    def test_average_neighbour_patterns_chunk(self, dummy_signal, dtype_in):
        w = kp.util.Window()

        # Get array to operate on
        dask_array = kp.util.dask._get_dask_array(dummy_signal)
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
            func=kp.util.chunk.average_neighbour_patterns,
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
    pass


class TestFFTFilterChunk:
    pass


class TestNormalizeIntensityChunk:
    pass

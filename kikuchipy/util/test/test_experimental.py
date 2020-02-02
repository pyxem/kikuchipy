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

import dask.array as da
import numpy as np
import pytest
from scipy.ndimage import convolve

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
        "kernel, kernel_size, answer, match, error_type, kwargs",
        [
            # Standard circular kernel
            (
                "circular",
                (3, 3),
                # fmt: off
                np.array(
                    [
                        [0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0],
                    ],
                ),
                # fmt: on
                None,
                None,
                None,
            ),
            # Circular kernel with first dimension even
            (
                "circular",
                (2, 3),
                # fmt: off
                np.array(
                    [
                        [0, 1, 0],
                        [1, 1, 1],
                    ],
                ),
                # fmt: on
                None,
                None,
                None,
            ),
            # Circular kernel with second dimension even
            (
                "circular",
                (3, 2),
                # fmt: off
                np.array(
                    [
                        [0, 1],
                        [1, 1],
                        [0, 1],
                    ],
                ),
                # fmt: on
                None,
                None,
                None,
            ),
            # Rectangular kernel
            ("rectangular", (2, 2), np.ones((2, 2)), None, None, None),
            # One keyword argument to scipy.signal.windows.get_window
            (
                "gaussian",
                (3, 3),
                # fmt: off
                np.array(
                    [
                        [0.77880078, 0.8824969, 0.77880078],
                        [0.8824969, 1., 0.8824969],
                        [0.77880078, 0.8824969, 0.77880078]
                    ]
                ),
                # fmt: on
                None,
                None,
                {"std": 2},
            ),
            # Two keyword arguments to scipy.signal.windows.get_window
            (
                "general_gaussian",
                (3, 2),
                # fmt: off
                np.array(
                    [
                        [0.96734205, 0.96734205],
                        [0.99804878, 0.99804878],
                        [0.96734205, 0.96734205]
                    ],
                ),
                # fmt: on
                None,
                None,
                {"sig": 2, "p": 2},
            ),
            # Integer kernel size
            ("rectangular", 3, np.ones(3), None, None, None),
            # Custom kernel
            (
                np.arange(9).reshape((3, 3)),
                (4, 2),  # Kernel size shouldn't matter with custom kernel
                np.arange(9).reshape((3, 3)),
                None,
                None,
                None,
            ),
            # Invalid custom kernel
            (
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                (9,),
                np.arange(9),
                "Kernel must be of type numpy.ndarray, however a kernel of ",
                ValueError,
                None,
            ),
            # Negative number as invalid kernel dimensions
            (
                "circular",
                (3, -3),
                # fmt: off
                np.array(
                    [
                        [0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0],
                    ],
                ),
                # fmt: on
                "Kernel dimensions must be positive, however .* was passed.",
                ValueError,
                None,
            ),
            # String as invalid kernel dimensions
            (
                "circular",
                "(3, -3)",
                # fmt: off
                np.array(
                    [
                        [0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0],
                    ],
                ),
                # fmt: on
                "Kernel dimensions must be an int or a tuple of ints, however ",
                TypeError,
                None,
            ),
            # Greater kernel dimension than scan dimension as invalid kernel
            # dimensions
            (
                "rectangular",
                (3, 4),
                np.ones(12).reshape((3, 4)),
                "Kernel size .* is too large for a scan of dimensions .*",
                ValueError,
                None,
            ),
        ],
    )
    def test_get_pattern_kernel(
        self,
        dummy_signal,
        kernel,
        kernel_size,
        answer,
        match,
        error_type,
        kwargs,
    ):
        if match is None:
            if kwargs is None:
                kernel = kp.util.experimental.get_pattern_kernel(
                    kernel=kernel,
                    kernel_size=kernel_size,
                    axes=dummy_signal.axes_manager,
                )
            else:
                kernel = kp.util.experimental.get_pattern_kernel(
                    kernel=kernel,
                    kernel_size=kernel_size,
                    axes=dummy_signal.axes_manager,
                    **kwargs,
                )
            np.testing.assert_array_almost_equal(kernel, answer)
        else:
            with pytest.raises(error_type, match=match):
                kp.util.experimental.get_pattern_kernel(
                    kernel=kernel,
                    kernel_size=kernel_size,
                    axes=dummy_signal.axes_manager,
                )

    def test_get_pattern_kernel_warns_kernel_dimensions(self, dummy_signal):
        with pytest.warns(
            UserWarning,
            match="Creates kernel of size .*, since input kernel size .* has",
        ):
            kp.util.experimental.get_pattern_kernel(
                kernel_size=(3, 3, 3), axes=dummy_signal.axes_manager
            )

    def test_get_pattern_kernel_invalid_axes_manager(self):
        with pytest.raises(AttributeError, match="A hyperspy.axes.AxesManager"):
            kp.util.experimental.get_pattern_kernel("circular", (3, 3), axes=1)

    @pytest.mark.parametrize("dtype_in", [None, np.uint8])
    def test_average_neighbour_patterns_chunk(self, dummy_signal, dtype_in):
        # Get averaging kernel
        kernel = kp.util.experimental.get_pattern_kernel(
            axes=dummy_signal.axes_manager,
        )
        expanded_kernel = kernel.reshape(
            kernel.shape + (1,) * dummy_signal.axes_manager.signal_dimension
        )

        # Get array to operate on
        dask_array = kp.util.dask._get_dask_array(dummy_signal)
        dtype_out = dask_array.dtype

        # Get sum of kernel coefficients for each pattern
        kernel_sums = convolve(
            input=np.ones(dummy_signal.axes_manager.navigation_shape[::-1]),
            weights=kernel,
            mode="constant",
            cval=0,
        )
        kernel_sums_expanded = da.from_array(
            kernel_sums.reshape(
                kernel_sums.shape
                + (1,) * dummy_signal.axes_manager.signal_dimension
            ),
            chunks=dask_array.chunksize,
        )

        averaged_patterns = dask_array.map_blocks(
            kp.util.experimental._average_neighbour_patterns_chunk,
            kernel_sums=kernel_sums_expanded,
            kernel=expanded_kernel,
            dtype_out=dtype_in,
            dtype=dtype_out,
        )

        answer = np.array([7, 4, 6, 6, 3, 7, 7, 3, 2], dtype=np.uint8).reshape(
            (3, 3)
        )

        # Check for correct data type and gives expected output intensities
        assert averaged_patterns.dtype == dtype_out
        np.testing.assert_almost_equal(
            averaged_patterns[0, 0].compute(), answer
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

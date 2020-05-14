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

import numpy as np
import pytest

from kikuchipy.filters.window import Window
from kikuchipy.pattern._pattern import (
    fft,
    fft_frequency_vectors,
    fft_spectrum,
    get_dynamic_background,
    get_image_quality,
    ifft,
    normalize_intensity,
    rescale_intensity,
    remove_dynamic_background,
    _rescale,
    _dynamic_background_frequency_space_setup,
)

# Expected output intensities from various image processing methods
RESCALED_UINT8 = np.array(
    [[182, 218, 182], [255, 218, 182], [218, 36, 0]], dtype=np.uint8
)
RESCALED_FLOAT32 = np.array(
    [[0.4285, 0.7142, 0.4285], [1, 0.7142, 0.4285], [0.7142, -0.7142, -1]],
    dtype=np.float32,
)
RESCALED_UINT8_0100 = np.array(
    [[71, 85, 71], [100, 85, 71], [85, 14, 0]], dtype=np.uint8
)
STATIC_CORR_UINT8 = np.array(
    [[0, 2, 0], [3, 3, 1], [2, 255, 255]], dtype=np.uint8
)
DYN_CORR_UINT8_SPATIAL_STD2 = np.array(
    [[170, 215, 181], [255, 221, 188], [221, 32, 0]], dtype=np.uint8
)
DYN_CORR_UINT8_SPATIAL_STD1 = np.array(
    [[120, 197, 156], [255, 241, 223], [226, 0, 9]], dtype=np.uint8
)
DYN_CORR_FLOAT32_SPATIAL_DIV_STD0375 = np.array(
    [[0.9248, 0.9727, 0.9449], [0.9865, 1, 0.9986], [0.9903, 0.5766, -1]],
    dtype=np.float32,
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


class TestRescaleIntensityPattern:
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
    def test_rescale_intensity(
        self, dummy_signal, dtype_out, out_range, answer
    ):
        pattern = dummy_signal.inav[0, 0].data

        # Check for accepted data types
        if dtype_out == np.complex:
            with pytest.raises(KeyError, match="Could not set output"):
                _ = rescale_intensity(
                    pattern=pattern,
                    in_range=None,
                    out_range=out_range,
                    dtype_out=dtype_out,
                )
            return 0  # So that the tests ends here
        else:
            rescaled_pattern = rescale_intensity(
                pattern=pattern,
                in_range=None,
                out_range=out_range,
                dtype_out=dtype_out,
            )

        # Check for correct data type and gives expected output intensities
        if dtype_out is not None:
            assert rescaled_pattern.dtype == dtype_out

        assert np.allclose(rescaled_pattern, answer, atol=1e-4)

    def test_rescale_intensity_py_func(self, dummy_signal):
        p = dummy_signal.inav[0, 0].data.astype(np.float32)
        imin, imax = np.min(p), np.max(p)
        omin, omax = -3, 300.15
        p2 = _rescale.py_func(
            pattern=p, imin=imin, imax=imax, omin=omin, omax=omax,
        )

        assert np.allclose(np.min(p2), omin)
        assert np.allclose(np.max(p2), omax)


class TestRemoveDynamicBackgroundPattern:
    @pytest.mark.parametrize(
        "std, operation, dtype_out, answer",
        [
            (1, "subtract", np.uint8, DYN_CORR_UINT8_SPATIAL_STD1),
            (2, "subtract", np.uint8, DYN_CORR_UINT8_SPATIAL_STD2),
            (None, "divide", np.float32, DYN_CORR_FLOAT32_SPATIAL_DIV_STD0375),
        ],
    )
    def test_remove_dynamic_background_spatial(
        self, dummy_signal, std, operation, dtype_out, answer
    ):
        p = dummy_signal.inav[0, 0].data.astype(np.float32)

        p2 = remove_dynamic_background(
            pattern=p,
            operation=operation,
            filter_domain="spatial",
            std=std,
            dtype_out=dtype_out,  # np.dtype("uint8").type
        )

        assert np.allclose(p2, answer, atol=1e-4)

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
        p = dummy_signal.inav[0, 0].data.astype(np.float32)

        p2 = remove_dynamic_background(
            pattern=p,
            operation="subtract",
            filter_domain="frequency",
            std=std,
            truncate=truncate,
            dtype_out=np.uint8,
        )

        assert np.allclose(p2, answer)

    def test_remove_dynamic_background_pattern_raises(self, dummy_signal):
        p = dummy_signal.inav[0, 0].data
        filter_domain = "Taldorei"
        with pytest.raises(ValueError, match=f"{filter_domain} must be "):
            _ = remove_dynamic_background(
                pattern=p, filter_domain=filter_domain,
            )

    def test_remove_dynamic_background_pattern_frequency_setup(
        self, dummy_signal
    ):
        std = 2
        truncate = 3.0

        (
            fft_shape,
            window_shape,
            window_fft,
            offset_before_fft,
            offset_after_ifft,
        ) = _dynamic_background_frequency_space_setup(
            pattern_shape=dummy_signal.axes_manager.signal_shape[::-1],
            std=std,
            truncate=truncate,
        )

        assert fft_shape == (8, 8)
        assert window_shape == (6, 6)
        assert np.sum(window_fft.imag) != 0
        assert offset_before_fft == (3, 3)
        assert offset_after_ifft == (2, 2)


class TestGetDynamicBackgroundPattern:
    @pytest.mark.parametrize(
        "std, truncate, answer",
        [
            (1, 4, np.array([[4, 4, 4], [5, 4, 3], [4, 2, 1]], dtype=np.uint8)),
            (2, 2, np.array([[4, 4, 3], [4, 4, 4], [4, 4, 4]], dtype=np.uint8)),
            (
                None,
                4,
                np.array([[4, 4, 4], [5, 4, 4], [5, 1, 0]], dtype=np.uint8),
            ),
        ],
    )
    def test_get_dynamic_background_pattern_spatial(
        self, dummy_signal, std, truncate, answer
    ):
        p = dummy_signal.inav[0, 0].data
        bg = get_dynamic_background(
            pattern=p, filter_domain="spatial", std=std, truncate=truncate,
        )

        assert np.allclose(bg, answer)

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
    def test_get_dynamic_background_frequency(self, dummy_signal, std, answer):

        p = dummy_signal.inav[0, 0].data.astype(answer.dtype)

        bg = get_dynamic_background(
            pattern=p, filter_domain="frequency", std=std,
        )

        assert np.allclose(bg, answer, atol=1e-4)

    def test_get_dynamic_background_raises(self, dummy_signal):
        p = dummy_signal.inav[0, 0].data
        filter_domain = "emon"
        with pytest.raises(ValueError, match=f"{filter_domain} must be either"):
            _ = get_dynamic_background(pattern=p, filter_domain=filter_domain)


class TestGetImageQuality:
    @pytest.mark.parametrize(
        "idx, normalize, frequency_vectors, inertia_max, answer",
        [
            ((0, 0), True, None, None, -0.0241),
            ((0, 0), False, None, None, 0.2694),
            ((2, 2), True, None, None, -0.2385),
        ],
    )
    def test_get_image_quality_pattern(
        self,
        dummy_signal,
        idx,
        normalize,
        frequency_vectors,
        inertia_max,
        answer,
    ):
        p = dummy_signal.inav[idx].data.astype(np.float32)
        iq = get_image_quality(
            pattern=p,
            normalize=normalize,
            frequency_vectors=frequency_vectors,
            inertia_max=inertia_max,
        )

        assert np.allclose(iq, answer, atol=1e-4)

    def test_get_image_quality_white_noise(self):
        p = np.random.random((1001, 1001))
        iq = get_image_quality(pattern=p, normalize=True)

        assert np.allclose(iq, 0, atol=1e-2)

    def test_get_image_quality_flat(self):
        p = np.ones((1001, 1001)) * 5
        iq = get_image_quality(pattern=p, normalize=False)

        assert np.allclose(iq, 1, atol=1e-2)

    @pytest.mark.parametrize(
        "shape, answer",
        [
            ((3, 3), np.array([[1, 4, 1], [4, 7, 4], [1, 4, 1]])),
            (
                (5, 4),
                # fmt: off
                np.array(
                    [
                        [1, 4, 4, 1],
                        [4, 7, 7, 4],
                        [9, 12, 12, 9],
                        [4, 7, 7, 4],
                        [1, 4, 4, 1],
                    ]
                ),
                # fmt: on
            ),
        ],
    )
    def test_fft_frequency_vectors(self, shape, answer):
        vec = fft_frequency_vectors(shape=shape)

        assert np.allclose(vec, answer)


class TestFFTPattern:
    @pytest.mark.parametrize(
        "shift, real_fft_only, expected_spectrum_sum",
        [
            (True, True, 15352),
            (True, False, 20402),
            (False, False, 20402),
            (False, True, 15352),
        ],
    )
    def test_fft_pattern(self, shift, real_fft_only, expected_spectrum_sum):
        p = np.ones((101, 101))
        p[50, 50] = 2

        kwargs = {}

        p_fft = fft(
            pattern=p, shift=shift, real_fft_only=real_fft_only, **kwargs,
        )

        assert np.allclose(
            np.sum(fft_spectrum.py_func(p_fft)),
            expected_spectrum_sum,
            atol=1e-3,
        )

    @pytest.mark.parametrize(
        "window", ["modified_hann", "tukey", "hamming"],
    )
    def test_fft_pattern_apodization_window(self, dummy_signal, window):
        p = dummy_signal.inav[0, 0].data
        w = Window(window, shape=p.shape)
        p2 = fft(pattern=p, apodization_window=w, shift=True,)
        p3 = fft(pattern=p * w, shift=True)
        p4 = fft(pattern=p, shift=True)

        assert p2.shape == p.shape
        assert p3.shape == p.shape
        assert np.allclose(p2, p3)
        assert not np.allclose(p2, p4, atol=1e-1)
        assert not np.allclose(p3, p4, atol=1e-1)

    @pytest.mark.parametrize("shift", [True, False])
    def test_ifft_pattern(self, shift):
        p = np.random.random((101, 101))
        p_fft = fft(p, shift=shift)
        p_ifft = ifft(p_fft, shift=shift)

        assert np.allclose(p_ifft, p)

    @pytest.mark.parametrize("shift", [True, False])
    def test_ifft_pattern_real(self, shift):
        # Odd second dimension becomes even with only real valued FFT
        p = np.random.random((101, 100))
        p_fft = fft(p, shift=shift, real_fft_only=False)
        p_ifft = ifft(p_fft, shift=shift, real_fft_only=False)

        p_rfft = fft(p, shift=shift, real_fft_only=True)
        p_irfft = ifft(p_rfft, shift=shift, real_fft_only=True)

        assert p_ifft.shape == p.shape
        assert p_irfft.shape == p.shape
        assert np.allclose(p_ifft, p_irfft)


class TestNormalizeIntensityPattern:
    @pytest.mark.parametrize(
        "num_std, divide_by_square_root, answer",
        [
            (
                1,
                True,
                # fmt: off
                np.array([
                    [0.0653, 0.2124, 0.0653],
                    [0.3595, 0.2124, 0.0653],
                    [0.2124, -0.5229, -0.6700],
                ])
                # fmt: on
            ),
            (
                2,
                True,
                # fmt: off
                np.array([
                    [0.0326, 0.1062, 0.0326],
                    [0.1797, 0.1062, 0.0326],
                    [0.1062, -0.2614, -0.3350],
                ]),
                # fmt: on
            ),
            (
                1,
                False,
                # fmt: off
                np.array([
                    [0.1961, 0.6373, 0.1961],
                    [1.0786, 0.6373, 0.1961],
                    [0.6373, -1.5689, -2.0101],
                ]),
                # fmt: on
            ),
        ],
    )
    def test_normalize_intensity_pattern(
        self, dummy_signal, num_std, divide_by_square_root, answer
    ):
        p = dummy_signal.inav[0, 0].data.astype(np.float32)
        p2 = normalize_intensity.py_func(
            pattern=p,
            num_std=num_std,
            divide_by_square_root=divide_by_square_root,
        )

        assert np.allclose(np.mean(p2), 0, atol=1e-6)
        assert np.allclose(p2, answer, atol=1e-4)

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

"""
This module contains tools to use the 2D image filter of numpy arrays
via FFT written by Connelly Barnes (public domain, 2007).
"""

from typing import Union, Tuple, List

import numpy as np
import scipy

from kikuchipy.util.window import Window


def _fft_filter_setup(
    image_shape: Tuple[int, ...], window: Union[np.ndarray, Window],
) -> Tuple[Tuple[int, ...], np.ndarray, Tuple[int, ...], Tuple[int, ...]]:
    # Optimal FFT shape
    image_shape = np.array(image_shape)
    window_shape = np.array(window.shape)
    fft_shape = tuple(
        [scipy.fft.next_fast_len(i) for i in (image_shape + window_shape - 1)]
    )

    # Pad window to optimal FFT size
    window_pad = _pad_window(window, fft_shape)

    # Compute real valued FFT of window
    window_rfft = scipy.fft.rfft2(window_pad)

    # Image offset before FFT and after IFFT
    offset_before = _offset_before_fft(window_shape)
    offset_after = _offset_after_ifft(window_shape)

    return fft_shape, window_rfft, offset_before, offset_after


def fft_filter(
    image: np.ndarray, window: Union[np.ndarray, Window],
) -> np.ndarray:
    """Filter a 2D image with a given window in the frequency domain.

    This method is based on the work by Connelly Barnes.

    Parameters
    ----------
    image
        Image to filter.
    window
        Kernel to filter the image with.

    Returns
    -------
    filtered_image
        Filtered image.

    """

    # Get optimal FFT shape, window padded with zeros in the optimal FFT
    # shape, real valued FFT of padded window, and image offsets before
    # FFT and after IFFT
    (
        fft_shape,
        kernel_fft,
        offset_before_fft,
        offset_after_ifft,
    ) = _fft_filter_setup(image.shape, window)

    # Pad image to optimal FFT shape and fill necessary parts, compute
    # the real valued FFT of the image, multiply FFT of image with FFT
    # of window, compute the inverse FFT, and finally remove padding
    filtered_image = _fft_filter(
        image,
        fft_shape,
        window.shape,
        kernel_fft,
        offset_before_fft,
        offset_after_ifft,
    )

    return filtered_image


def _pad_window(
    window: Union[np.ndarray, Window],
    fft_shape: Union[Tuple[int, ...], List[int]],
) -> np.ndarray:
    ky, kx = window.shape
    window_pad = np.zeros(fft_shape, dtype=np.float32)
    window_pad[0:ky, 0:kx] = np.flipud(np.fliplr(window))
    return window_pad


def _offset_before_fft(
    window_shape: Union[Tuple[int, int], np.ndarray]
) -> Tuple[int, int]:
    ky, kx = window_shape
    offset = (kx - ((kx - 1) // 2) - 1, ky - ((ky - 1) // 2) - 1)
    return offset


def _offset_after_ifft(
    window_shape: Union[Tuple[int, int], np.ndarray]
) -> Tuple[int, int]:
    ky, kx = window_shape
    offset = ((ky - 1) // 2, (kx - 1) // 2)
    return offset


def _pad_image(image, fft_shape, window_shape, offset_before_fft):
    iy, ix = image.shape
    ky, kx = window_shape
    fy, fx = fft_shape
    oy, ox = offset_before_fft

    image_pad = np.zeros(fft_shape, dtype=np.float32)
    image_pad[0:iy, 0:ix] = image

    # Pad new image array
    image_pad[iy : iy + (ky - 1) // 2, :ix] = image[iy - 1, :]
    image_pad[:iy, ix : ix + (kx - 1) // 2] = np.expand_dims(
        image[:, ix - 1], axis=1
    )
    image_pad[fy - ox :, :ix] = image[0, :]
    image_pad[:iy, fx - oy :] = np.expand_dims(image[:, 0], axis=1)
    image_pad[iy : iy + (ky - 1) // 2, ix : ix + (kx - 1) // 2] = image[-1, -1]
    image_pad[fy - ox :, ix : ix + (kx - 1) // 2] = image[0, -1]
    image_pad[iy : iy + (ky - 1) // 2, fx - oy :] = image[-1, 0]
    image_pad[fy - ox :, fx - oy :] = image[0, 0]

    return image_pad


def _fft_filter(
    image: np.ndarray,
    fft_shape: Union[Tuple[int, ...]],
    window_shape: Union[List[int], Tuple[int, ...]],
    window_fft: np.ndarray,
    offset_before_fft: Tuple[int, ...],
    offset_after_ifft: Tuple[int, ...],
):
    # Create new image array to pad with the image in the top left
    # corner
    image_pad = _pad_image(image, fft_shape, window_shape, offset_before_fft)

    # Compute real valued FFT of image and filter
    image_pad_fft = scipy.fft.rfft2(image_pad)

    # Compute inverse FFT of product between FFTs
    result_fft = scipy.fft.irfft2(image_pad_fft * window_fft, fft_shape)

    # Return filtered image without padding
    iy, ix = image.shape
    oy2, ox2 = offset_after_ifft

    return result_fft[oy2 : oy2 + iy, ox2 : ox2 + ix]

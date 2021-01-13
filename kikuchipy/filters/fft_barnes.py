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

"""
This module contains tools to use the 2D image filter of numpy arrays
via FFT written by Connelly Barnes (public domain, 2007).
"""

from typing import Union, Tuple, List

import numpy as np
from numpy.fft import rfft2, irfft2
from scipy.fftpack import next_fast_len

# from scipy.fft import next_fast_len, rfft2, irfft2

from kikuchipy.filters.window import Window


def _fft_filter_setup(
    image_shape: Tuple[int, int], window: Union[np.ndarray, Window],
) -> Tuple[Tuple[int, int], np.ndarray, Tuple[int, int], Tuple[int, int]]:
    window_shape = window.shape

    # Optimal FFT shape
    #    real_fft_only = True
    fft_shape = (
        next_fast_len(
            image_shape[0] + window_shape[0] - 1
        ),  # , real_fft_only),
        next_fast_len(
            image_shape[1] + window_shape[1] - 1
        ),  # , real_fft_only),
    )

    # Pad window to optimal FFT size
    window_pad = _pad_window(window, fft_shape)

    # Obtain the transfer function via the real valued FFT
    transfer_function = rfft2(window_pad)

    # Image offset before FFT and after IFFT
    offset_before = _offset_before_fft(window_shape)
    offset_after = _offset_after_ifft(window_shape)

    return fft_shape, transfer_function, offset_before, offset_after


def fft_filter(
    image: np.ndarray, window: Union[np.ndarray, Window],
) -> np.ndarray:
    """Filter a 2D image in the frequency domain with a window defined
    in the spatial domain.

    This method is based on the work by Connelly Barnes.

    Parameters
    ----------
    image
        Image to filter.
    window
        Window to filter the image with defined in the spatial domain.

    Returns
    -------
    filtered_image
        Filtered image.
    """
    # Get optimal FFT shape, window padded with zeros in the optimal FFT
    # shape, transfer function of padded window, and image offsets
    # before FFT and after IFFT
    (
        fft_shape,
        transfer_function,
        offset_before_fft,
        offset_after_ifft,
    ) = _fft_filter_setup(image.shape, window)

    # Pad image to optimal FFT shape and fill necessary parts, compute
    # the real valued FFT of the image, multiply FFT of image with FFT
    # of window, compute the inverse FFT, and finally remove padding
    filtered_image = _fft_filter(
        image=image,
        fft_shape=fft_shape,
        window_shape=window.shape,
        transfer_function=transfer_function,
        offset_before_fft=offset_before_fft,
        offset_after_ifft=offset_after_ifft,
    )

    return filtered_image


def _pad_window(
    window: Union[np.ndarray, Window],
    fft_shape: Union[Tuple[int, ...], List[int], np.ndarray],
) -> np.ndarray:
    wy, wx = window.shape
    window_pad = np.zeros(fft_shape, dtype=np.float32)
    window_pad[:wy, :wx] = np.flipud(np.fliplr(window))

    return window_pad


def _offset_before_fft(
    window_shape: Union[Tuple[int, int], np.ndarray]
) -> Tuple[int, int]:
    wy, wx = window_shape
    offset = (wy - ((wy - 1) // 2) - 1, wx - ((wx - 1) // 2) - 1)
    return offset


def _offset_after_ifft(
    window_shape: Union[Tuple[int, int], np.ndarray]
) -> Tuple[int, int]:
    wy, wx = window_shape
    offset = ((wy - 1) // 2, (wx - 1) // 2)
    return offset


def _pad_image(
    image: np.ndarray,
    fft_shape: Tuple[int, ...],
    window_shape: Tuple[int, int],
    offset_before_fft: Tuple[int, int],
) -> np.ndarray:
    iy, ix = image.shape
    wy, wx = window_shape
    fy, fx = fft_shape
    oy, ox = offset_before_fft

    image_pad = np.zeros(fft_shape, dtype=np.float32)
    image_pad[0:iy, 0:ix] = image

    # Pad new image array:
    # Extend bottom row below
    image_pad[iy : iy + (wy - 1) // 2, :ix] = image[-1, :]
    # Extend right most column to the right
    image_pad[:iy, ix : ix + (wx - 1) // 2] = np.expand_dims(
        image[:, -1], axis=1
    )
    # Pad top row below
    image_pad[fy - oy :, :ix] = image[0, :]
    # Pad left most column to the right
    image_pad[:iy, fx - ox :] = np.expand_dims(image[:, 0], axis=1)
    # Extend bottom right corner to a square below/right
    image_pad[iy : iy + (wy - 1) // 2, ix : ix + (wx - 1) // 2] = image[-1, -1]
    # Extend upper right corner to a square bottom/right
    image_pad[fy - oy :, ix : ix + (wx - 1) // 2] = image[0, -1]
    # Extend bottom left corner to a square right/below
    image_pad[iy : iy + (wy - 1) // 2, fx - ox :] = image[-1, 0]
    # Extend upper left corner to a square bottom/right
    image_pad[fy - oy :, fx - ox :] = image[0, 0]

    return image_pad


def _fft_filter(
    image: np.ndarray,
    transfer_function: np.ndarray,
    fft_shape: Tuple[int, int],
    window_shape: Tuple[int, int],
    offset_before_fft: Tuple[int, int],
    offset_after_ifft: Tuple[int, int],
) -> np.ndarray:
    # Create new image array to pad with the image in the top left
    # corner
    image_pad = _pad_image(image, fft_shape, window_shape, offset_before_fft)

    # Compute real valued FFT of image
    image_pad_fft = rfft2(image_pad)

    # Compute inverse FFT of product between FFTs
    result_fft = irfft2(image_pad_fft * transfer_function, fft_shape)

    # Return filtered image without padding
    iy, ix = image.shape
    oy, ox = offset_after_ifft

    return np.real(result_fft[oy : oy + iy, ox : ox + ix])

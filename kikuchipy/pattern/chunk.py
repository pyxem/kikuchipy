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

"""Private functions for operating on :class:`numpy.ndarray` or
:class:`dask.array.Array` chunks of EBSD patterns.
"""

from typing import Union

import dask.array as da
from numba import njit
import numpy as np
from scipy.ndimage import correlate, gaussian_filter

import kikuchipy.pattern._pattern as pattern_processing
import kikuchipy.filters.fft_barnes as barnes
from kikuchipy.filters.window import Window
from kikuchipy.pattern._pattern import _rescale_with_min_max


def get_dynamic_background(
    patterns: Union[np.ndarray, da.Array],
    filter_func: Union[gaussian_filter, barnes.fft_filter],
    dtype_out: Union[str, np.dtype, type, None] = None,
    **kwargs,
) -> np.ndarray:
    """Obtain the dynamic background in a chunk of EBSD patterns.

    Parameters
    ----------
    patterns
        EBSD patterns.
    filter_func
        Function where a Gaussian convolution filter is applied, in the
        frequency or spatial domain. Either
        :func:`scipy.ndimage.gaussian_filter` or
        :func:`kikuchipy.util.barnes_fftfilter.fft_filter`.
    dtype_out
        Data type of background patterns. If None (default), it is set
        to input patterns' data type.
    **kwargs
        Keyword arguments passed to the Gaussian blurring function
        passed to `filter_func`.

    Returns
    -------
    background : numpy.ndarray
        Large scale variations in the input EBSD patterns.
    """
    if dtype_out is None:
        dtype_out = patterns.dtype
    else:
        dtype_out = np.dtype(dtype_out)

    background = np.empty_like(patterns, dtype=dtype_out)

    for nav_idx in np.ndindex(patterns.shape[:-2]):
        background[nav_idx] = filter_func(patterns[nav_idx], **kwargs)

    return background


def fft_filter(
    patterns: np.ndarray,
    filter_func: Union[pattern_processing.fft_filter, barnes._fft_filter],
    transfer_function: Union[np.ndarray, Window],
    dtype_out: Union[str, np.dtype, type, None] = None,
    **kwargs,
) -> np.ndarray:
    """Filter a chunk of EBSD patterns in the frequency domain.

    Patterns are transformed via the Fast Fourier Transform (FFT) to the
    frequency domain, where their spectrum is multiplied by a filter
    `transfer_function`, and the filtered spectrum is subsequently
    transformed to the spatial domain via the inverse FFT (IFFT).

    Filtered patterns are rescaled to the data type range of
    `dtype_out`.

    Parameters
    ----------
    patterns
        EBSD patterns.
    filter_func
        Function to apply `transfer_function` with.
    transfer_function
        Filter transfer function in the frequency domain.
    dtype_out
        Data type of output patterns. If None (default), it is set to
        the input patterns' data type.
    **kwargs
        Keyword arguments passed to the `filter_func`.

    Returns
    -------
    filtered_patterns
        Filtered EBSD patterns.
    """
    if dtype_out is None:
        dtype_out = patterns.dtype.type
    else:
        dtype_out = np.dtype(dtype_out)

    filtered_patterns = np.empty_like(patterns, dtype=dtype_out)

    for nav_idx in np.ndindex(patterns.shape[:-2]):
        filtered_pattern = filter_func(
            patterns[nav_idx], transfer_function=transfer_function, **kwargs
        )

        # Rescale the pattern intensity
        filtered_patterns[nav_idx] = pattern_processing.rescale_intensity(
            filtered_pattern, dtype_out=dtype_out
        )

    return filtered_patterns


def _average_neighbour_patterns(
    patterns: np.ndarray,
    window_sums: np.ndarray,
    window: Union[np.ndarray, Window],
    dtype_out: np.dtype,
    omin: float,
    omax: float,
) -> np.ndarray:
    """See docstring of :func:`average_neighbour_patterns`."""
    patterns = patterns.astype("float32")
    correlated_patterns = correlate(patterns, weights=window, mode="constant")
    rescaled_patterns = _rescale_neighbour_averaged_patterns(
        correlated_patterns, window_sums, dtype_out, omin, omax
    )
    return rescaled_patterns


@njit(cache=True, fastmath=True, nogil=True)
def _rescale_neighbour_averaged_patterns(
    patterns: np.ndarray,
    window_sums: np.ndarray,
    dtype_out: np.dtype,
    omin: float,
    omax: float,
) -> np.ndarray:
    """See docstring of :func:`average_neighbour_patterns`."""
    rescaled_patterns = np.zeros(patterns.shape, dtype=dtype_out)
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        pattern_i = patterns[nav_idx] / window_sums[nav_idx]
        imin = np.min(pattern_i)
        imax = np.max(pattern_i)
        rescaled_patterns[nav_idx] = _rescale_with_min_max(
            pattern_i, imin, imax, omin, omax
        )
    return rescaled_patterns

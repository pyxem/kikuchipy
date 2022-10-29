# Copyright 2019-2022 The kikuchipy developers
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

"""Functions for operating on :class:`numpy.ndarray` or
:class:`dask.array.Array` chunks of EBSD patterns.

.. warning::

    This module will be become private for internal use only in v0.7.
    If you need to process multiple EBSD patterns at once, please do
    this using :class:`~kikuchipy.signals.EBSD`.
"""

from typing import Union, Tuple, List

import dask.array as da
from numba import njit
import numpy as np
from scipy.ndimage import correlate, gaussian_filter
from skimage.exposure import equalize_adapthist

import kikuchipy.pattern._pattern as pattern_processing
import kikuchipy.filters.fft_barnes as barnes
from kikuchipy.filters.window import Window
from kikuchipy.pattern._pattern import _rescale_with_min_max


def rescale_intensity(
    patterns: Union[np.ndarray, da.Array],
    in_range: Union[None, Tuple[int, int], Tuple[float, float]] = None,
    out_range: Union[None, Tuple[int, int], Tuple[float, float]] = None,
    dtype_out: Union[
        str, np.dtype, type, Tuple[int, int], Tuple[float, float], None
    ] = None,
    percentiles: Union[None, Tuple[int, int], Tuple[float, float]] = None,
) -> Union[np.ndarray, da.Array]:
    """Rescale pattern intensities in a chunk of EBSD patterns.

    Chunk max./min. intensity is determined from `out_range` or the
    data type range of :class:`numpy.dtype` passed to `dtype_out`.

    Parameters
    ----------
    patterns
        EBSD patterns.
    in_range
        Min./max. intensity of input patterns. If None (default),
        `in_range` is set to pattern min./max. Contrast stretching is
        performed when `in_range` is set to a narrower intensity range
        than the input patterns.
    out_range
        Min./max. intensity of output patterns. If None (default),
        `out_range` is set to `dtype_out` min./max according to
        `skimage.util.dtype.dtype_range`.
    dtype_out
        Data type of rescaled patterns. If None (default), it is set to
        the same data type as the input patterns.
    percentiles
        Disregard intensities outside these percentiles. Calculated
        per pattern. Must be None if `in_range` is passed (default
        is None).

    Returns
    -------
    rescaled_patterns : numpy.ndarray
        Rescaled patterns.
    """
    dtype_out = np.dtype(dtype_out)
    rescaled_patterns = np.empty_like(patterns, dtype=dtype_out)

    for nav_idx in np.ndindex(patterns.shape[:-2]):
        pattern = patterns[nav_idx]

        if percentiles is not None:
            in_range = np.percentile(pattern, q=percentiles)

        rescaled_patterns[nav_idx] = pattern_processing.rescale_intensity(
            pattern=pattern,
            in_range=in_range,
            out_range=out_range,
            dtype_out=dtype_out,
        )

    return rescaled_patterns


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


def adaptive_histogram_equalization(
    patterns: Union[np.ndarray, da.Array],
    kernel_size: Union[Tuple[int, int], List[int]],
    clip_limit: Union[int, float] = 0,
    nbins: int = 128,
) -> np.ndarray:
    """Local contrast enhancement of a chunk of EBSD patterns with
    adaptive histogram equalization.

    This method makes use of :func:`skimage.exposure.equalize_adapthist`.

    Parameters
    ----------
    patterns
        EBSD patterns.
    kernel_size
        Shape of contextual regions for adaptive histogram equalization.
    clip_limit
        Clipping limit, normalized between 0 and 1 (higher values give
        more contrast). Default is 0.
    nbins
        Number of gray bins for histogram. Default is 128.

    Returns
    -------
    equalized_patterns : numpy.ndarray
        Patterns with enhanced contrast.
    """
    dtype_in = patterns.dtype.type

    equalized_patterns = np.empty_like(patterns)

    for nav_idx in np.ndindex(patterns.shape[:-2]):

        # Adaptive histogram equalization
        equalized_pattern = equalize_adapthist(
            patterns[nav_idx],
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins,
        )

        # Rescale intensities
        equalized_patterns[nav_idx] = pattern_processing.rescale_intensity(
            equalized_pattern, dtype_out=dtype_in
        )

    return equalized_patterns


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


def normalize_intensity(
    patterns: Union[np.ndarray, da.Array],
    num_std: int = 1,
    divide_by_square_root: bool = False,
    dtype_out: Union[str, np.dtype, type, None] = None,
) -> np.ndarray:
    """Normalize intensities in a chunk of EBSD patterns to a mean of
    zero with a given standard deviation.

    Parameters
    ----------
    patterns
        Patterns to normalize the intensity in.
    num_std
        Number of standard deviations of the output intensities. Default
        is 1.
    divide_by_square_root
        Whether to divide output intensities by the square root of the
        pattern size. Default is False.
    dtype_out
        Data type of normalized patterns. If None (default), the input
        patterns' data type is used.

    Returns
    -------
    normalized_patterns
        Normalized patterns.

    Notes
    -----
    Data type should always be changed to floating point, e.g.
    ``"float32"`` with :meth:`numpy.ndarray.astype`, before normalizing
    the intensities.
    """
    if dtype_out is None:
        dtype_out = patterns.dtype
    else:
        dtype_out = np.dtype(dtype_out)

    normalized_patterns = np.empty_like(patterns, dtype=dtype_out)

    for nav_idx in np.ndindex(patterns.shape[:-2]):
        normalized_patterns[nav_idx] = pattern_processing.normalize_intensity(
            pattern=patterns[nav_idx],
            num_std=num_std,
            divide_by_square_root=divide_by_square_root,
        )

    return normalized_patterns


def average_neighbour_patterns(
    patterns: np.ndarray,
    window_sums: np.ndarray,
    window: Union[np.ndarray, Window],
    dtype_out: Union[str, np.dtype, type, None] = None,
) -> np.ndarray:
    """Average a chunk of patterns with its neighbours within a window.

    The amount of averaging is specified by the window coefficients.
    All patterns are averaged with the same window. Map borders are
    extended with zeros. Resulting pattern intensities are rescaled
    to fill the input patterns' data type range individually.

    Parameters
    ----------
    patterns
        Patterns to average, with some overlap with surrounding chunks.
    window_sums
        Sum of window data for each image.
    window
        Averaging window.
    dtype_out
        Data type of averaged patterns. If None (default), it is set to
        the same data type as the input patterns.

    Returns
    -------
    averaged_patterns
        Averaged patterns.
    """
    if dtype_out is None:
        dtype_out = patterns.dtype
    else:
        dtype_out = np.dtype(dtype_out)

    # Correlate patterns with window
    correlated_patterns = correlate(patterns, weights=window, mode="constant", cval=0)

    # Divide convolved patterns by number of neighbours averaged with
    averaged_patterns = np.empty_like(correlated_patterns, dtype=dtype_out)
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        averaged_patterns[nav_idx] = pattern_processing.rescale_intensity(
            pattern=correlated_patterns[nav_idx] / window_sums[nav_idx],
            dtype_out=dtype_out,
        )

    return averaged_patterns


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

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

"""Functions for operating on :class:`numpy.ndarray` or
:class:`dask.array.Array` chunks of EBSD patterns.
"""

from typing import Union, Optional, Tuple, List

import dask.array as da
import numpy as np
from scipy.ndimage import correlate, gaussian_filter
from skimage.exposure import equalize_adapthist
from skimage.util.dtype import dtype_range

import kikuchipy.pattern._pattern as pattern_processing
import kikuchipy.filters.fft_barnes as barnes
from kikuchipy.filters.window import Window


def rescale_intensity(
    patterns: Union[np.ndarray, da.Array],
    in_range: Union[None, Tuple[int, int], Tuple[float, float]] = None,
    out_range: Union[None, Tuple[int, int], Tuple[float, float]] = None,
    dtype_out: Union[
        None, np.dtype, Tuple[int, int], Tuple[float, float]
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


def remove_static_background(
    patterns: Union[np.ndarray, da.Array],
    static_bg: Union[np.ndarray, da.Array],
    operation_func: Union[np.subtract, np.divide],
    scale_bg: bool = False,
    in_range: Union[None, Tuple[int, int], Tuple[float, float]] = None,
    out_range: Union[None, Tuple[int, int], Tuple[float, float]] = None,
    dtype_out: Union[
        None, np.dtype, Tuple[int, int], Tuple[float, float]
    ] = None,
) -> np.ndarray:
    """Remove the static background in a chunk of EBSD patterns.

    Removal is performed by subtracting or dividing by a static
    background pattern. Resulting pattern intensities are rescaled
    keeping relative intensities or not and stretched to fill the
    available grey levels in the patterns' data type range.

    Parameters
    ----------
    patterns
        EBSD patterns.
    static_bg
        Static background pattern. If None is passed (default) we try to
        read it from the signal metadata.
    operation_func
        Function to subtract or divide by the dynamic background
        pattern.
    scale_bg
        Whether to scale the static background pattern to each
        individual pattern's data range before removal (default is
        False).
    in_range
        Min./max. intensity values of input and output patterns. If None
        (default), it is set to the overall pattern min./max, losing
        relative intensities between patterns.
    out_range
        Min./max. intensity values of the output patterns. If None
        (default), `out_range` is set to `dtype_out` min./max according
        to `skimage.util.dtype.dtype_range`.
    dtype_out
        Data type of corrected patterns. If None (default), it is set to
        input patterns' data type.

    Returns
    -------
    corrected_patterns : numpy.ndarray
        Patterns with the static background removed.
    """
    if dtype_out is None:
        dtype_out = patterns.dtype.type

    if out_range is None:
        out_range = dtype_range[dtype_out]

    corrected_patterns = np.empty_like(patterns, dtype=dtype_out)

    for nav_idx in np.ndindex(patterns.shape[:-2]):
        # Get pattern
        pattern = patterns[nav_idx]

        # Scale background
        new_static_bg = static_bg
        if scale_bg:
            new_static_bg = pattern_processing.rescale_intensity(
                pattern=static_bg, out_range=(np.min(pattern), np.max(pattern))
            )

        # Remove the static background
        corrected_pattern = operation_func(pattern, new_static_bg)

        # Rescale the intensities
        corrected_patterns[nav_idx] = pattern_processing.rescale_intensity(
            pattern=corrected_pattern,
            in_range=in_range,
            out_range=out_range,
            dtype_out=dtype_out,
        )

    return corrected_patterns


def get_dynamic_background(
    patterns: Union[np.ndarray, da.Array],
    filter_func: Union[gaussian_filter, barnes.fft_filter],
    dtype_out: Union[
        None, np.dtype, Tuple[int, int], Tuple[float, float]
    ] = None,
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
    kwargs :
        Keyword arguments passed to the Gaussian blurring function
        passed to `filter_func`.

    Returns
    -------
    background : numpy.ndarray
        Large scale variations in the input EBSD patterns.
    """
    if dtype_out is None:
        dtype_out = patterns.dtype.type

    background = np.empty_like(patterns, dtype=dtype_out)

    for nav_idx in np.ndindex(patterns.shape[:-2]):
        background[nav_idx] = filter_func(patterns[nav_idx], **kwargs)

    return background


def remove_dynamic_background(
    patterns: Union[np.ndarray, da.Array],
    filter_func: Union[gaussian_filter, barnes.fft_filter],
    operation_func: Union[np.subtract, np.divide],
    out_range: Union[None, Tuple[int, int], Tuple[float, float]] = None,
    dtype_out: Union[
        None, np.dtype, Tuple[int, int], Tuple[float, float]
    ] = None,
    **kwargs,
) -> np.ndarray:
    """Correct the dynamic background in a chunk of EBSD patterns.

    The correction is performed by subtracting or dividing by a Gaussian
    blurred version of each pattern. Returned pattern intensities are
    rescaled to fill the input data type range.

    Parameters
    ----------
    patterns
        EBSD patterns.
    filter_func
        Function where a Gaussian convolution filter is applied, in the
        frequency or spatial domain. Either
        :func:`scipy.ndimage.gaussian_filter` or
        :func:`kikuchipy.util.barnes_fftfilter.fft_filter`.
    operation_func
        Function to subtract or divide by the dynamic background
        pattern.
    out_range
        Min./max. intensity values of the output patterns. If None
        (default), `out_range` is set to `dtype_out` min./max according
        to `skimage.util.dtype.dtype_range`.
    dtype_out
        Data type of corrected patterns. If None (default), it is set to
        input patterns' data type.
    kwargs :
        Keyword arguments passed to the Gaussian blurring function
        passed to `filter_func`.

    Returns
    -------
    corrected_patterns : numpy.ndarray
        Dynamic background corrected patterns.

    See Also
    --------
    kikuchipy.signals.ebsd.EBSD.remove_dynamic_background
    kikuchipy.util.pattern.remove_dynamic_background
    """
    if dtype_out is None:
        dtype_out = patterns.dtype.type

    if out_range is None:
        out_range = dtype_range[dtype_out]

    corrected_patterns = np.empty_like(patterns, dtype=dtype_out)

    for nav_idx in np.ndindex(patterns.shape[:-2]):
        # Get pattern
        pattern = patterns[nav_idx]

        # Get dynamic background by Gaussian filtering in frequency or
        # spatial domain
        dynamic_bg = filter_func(pattern, **kwargs)

        # Remove dynamic background
        corrected_pattern = operation_func(pattern, dynamic_bg)

        # Rescale intensities
        corrected_patterns[nav_idx] = pattern_processing.rescale_intensity(
            pattern=corrected_pattern, out_range=out_range, dtype_out=dtype_out,
        )

    return corrected_patterns


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


def get_image_quality(
    patterns: Union[np.ndarray, da.Array],
    frequency_vectors: Optional[np.ndarray] = None,
    inertia_max: Union[None, int, float] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute the image quality in a chunk of EBSD patterns.

    The image quality is calculated based on the procedure defined by
    Krieger Lassen [Lassen1994]_.

    Parameters
    ----------
    patterns
        EBSD patterns.
    frequency_vectors
        Integer 2D array with values corresponding to the weight given
        each FFT spectrum frequency component. If None (default), these
        are calculated from
        :func:`~kikuchipy.util.pattern.fft_frequency_vectors`.
    inertia_max
        Maximum inertia of the FFT power spectrum of the image. If None
        (default), this is calculated from the `frequency_vectors`.
    normalize
        Whether to normalize patterns to a mean of zero and standard
        deviation of 1 before calculating the image quality. Default
        is True.

    Returns
    -------
    image_quality_chunk : numpy.ndarray
        Image quality of patterns.
    """
    dtype_out = np.float64

    image_quality_chunk = np.empty(patterns.shape[:-2], dtype=dtype_out)

    for nav_idx in np.ndindex(patterns.shape[:-2]):
        # Get (normalized) pattern
        if normalize:
            pattern = pattern_processing.normalize_intensity(
                pattern=patterns[nav_idx]
            )
        else:
            pattern = patterns[nav_idx]

        # Compute image quality
        image_quality_chunk[nav_idx] = pattern_processing.get_image_quality(
            pattern=pattern,
            normalize=False,
            frequency_vectors=frequency_vectors,
            inertia_max=inertia_max,
        )

    return image_quality_chunk


def fft_filter(
    patterns: np.ndarray,
    filter_func: Union[pattern_processing.fft_filter, barnes._fft_filter],
    transfer_function: Union[np.ndarray, Window],
    dtype_out: Union[
        None, np.dtype, Tuple[int, int], Tuple[float, float]
    ] = None,
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
    kwargs :
        Keyword arguments passed to the `filter_func`.

    Returns
    -------
    filtered_patterns : numpy.ndarray
        Filtered EBSD patterns.
    """
    if dtype_out is None:
        dtype_out = patterns.dtype.type

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
    dtype_out: Optional[np.dtype] = None,
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
    normalized_patterns : numpy.ndarray
        Normalized patterns.

    Notes
    -----
    Data type should always be changed to floating point, e.g.
    ``np.float32`` with :meth:`numpy.ndarray.astype`, before normalizing
    the intensities.
    """
    if dtype_out is None:
        dtype_out = patterns.dtype.type

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
    dtype_out: Union[
        None, np.dtype, Tuple[int, int], Tuple[float, float]
    ] = None,
) -> np.ndarray:
    """Average a chunk of patterns with its neighbours within a window.

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
    averaged_patterns : numpy.ndarray
        Averaged patterns.
    """
    if dtype_out is None:
        dtype_out = patterns.dtype.type

    # Correlate patterns with window
    correlated_patterns = correlate(
        patterns.astype(np.float32), weights=window, mode="constant", cval=0,
    )

    # Divide convolved patterns by number of neighbours averaged with
    averaged_patterns = np.empty_like(correlated_patterns, dtype=dtype_out)
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        averaged_patterns[nav_idx] = (
            correlated_patterns[nav_idx] / window_sums[nav_idx]
        )

    return averaged_patterns

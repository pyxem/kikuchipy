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
from scipy.ndimage import gaussian_filter, convolve
from skimage.exposure import equalize_adapthist
from skimage.util.dtype import dtype_range


def _rescale_pattern(pattern, in_range=None, out_range=None, dtype_out=None):
    """Rescale pattern intensities inplace to desired
    :class:`numpy.dtype` range specified by ``dtype_out`` keeping
    relative intensities or not.

    This method makes use of :func:`skimage.exposure.rescale_intensity`.

    Parameters
    ----------
    pattern : dask.array.Array
        Pattern to rescale.
    in_range, out_range : tuple of int or float, optional
        Min./max. intensity values of input and output pattern. If None,
        (default) `in_range` is set to pattern min./max. If None
        (default), `out_range` is set to `dtype_out` min./max
        according to `skimage.util.dtype.dtype_range`, with min. equal
        to zero.
    dtype_out : np.dtype, optional
        Data type of rescaled pattern. If None (default), it is set to
        the same data type as the input pattern.

    Returns
    -------
    rescaled_pattern : da.Array
        Rescaled pattern.
    """

    if dtype_out is None:
        dtype_out = pattern.dtype

    if in_range is None:
        imin, imax = (pattern.min(), pattern.max())
    else:
        imin, imax = in_range
        pattern.clip(imin, imax)

    if out_range is None or out_range in dtype_range:
        omin = 0
        try:
            if isinstance(dtype_out, np.dtype):
                dtype_out = dtype_out.type
            _, omax = dtype_range[dtype_out]
        except KeyError:
            raise KeyError(
                "Could not set output intensity range, since data type "
                f"'{dtype_out}' is not recognised. Use any of '{dtype_range}'."
            )
    else:
        omin, omax = out_range

    rescaled_pattern = (pattern - imin) / float(imax - imin)

    return (rescaled_pattern * (omax - omin) + omin).astype(dtype_out)


def _rescale_pattern_chunk(
    patterns, in_range=None, out_range=None, dtype_out=None
):
    """Rescale patterns in chunk to fill the data type range using an
    approach inspired by `skimage.exposure.rescale_intensity`, keeping
    relative intensities or not.

    Parameters
    ----------
    patterns : da.Array
        Patterns to rescale.
    in_range, out_range : tuple of int or float, optional
        Min./max. intensity values of input and output pattern. If None
        (default), `in_range` is set to pattern min./max. If None
        (default), `out_range` is set to `dtype_out` min./max
        according to `skimage.util.dtype_out.dtype_range`, with min. equal
        to zero.
    dtype_out : np.dtype, optional
        Data type of rescaled patterns. If None (default), it is set to
        the same data type as the input patterns.

    Returns
    -------
    rescaled_patterns : da.Array
        Rescaled patterns.
    """

    rescaled_patterns = np.empty_like(patterns, dtype=dtype_out)
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        rescaled_patterns[nav_idx] = _rescale_pattern(
            patterns[nav_idx],
            in_range=in_range,
            out_range=out_range,
            dtype_out=dtype_out,
        )

    return rescaled_patterns


def _static_background_correction_chunk(
    patterns, static_bg, operation="subtract", in_range=None, dtype_out=None
):
    """Correct static background in patterns in chunk by subtracting or
    dividing by a static background pattern. Returned pattern
    intensities are rescaled keeping relative intensities or not and
    stretched to fill the input data type range.

    Parameters
    ----------
    patterns : da.Array
        Patterns to correct static background in.
    static_bg : np.ndarray or da.Array
        Static background pattern. If not passed we try to read it
        from the signal metadata.
    operation : 'subtract' or 'divide', optional
        Subtract (default) or divide by static background pattern.
    in_range : tuple of int or float, optional
        Min./max. intensity values of input and output patterns. If
        None, (default) `in_range` is set to pattern min./max, losing
        relative intensities between patterns.
    dtype_out : np.dtype, optional
        Data type of corrected patterns. If None (default), it is set to
        the same data type as the input patterns.

    Returns
    -------
    corrected_patterns : da.Array
        Static background corrected patterns.
    """

    if dtype_out is None:
        dtype_out = patterns.dtype

    corrected_patterns = np.empty_like(patterns, dtype=dtype_out)
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        if operation == "subtract":
            corrected_pattern = patterns[nav_idx] - static_bg
        else:  # Divide
            corrected_pattern = patterns[nav_idx] / static_bg
        corrected_patterns[nav_idx] = _rescale_pattern(
            corrected_pattern, in_range=in_range, dtype_out=dtype_out
        )

    return corrected_patterns


def _dynamic_background_correction_chunk(
    patterns, sigma, operation="subtract", dtype_out=None
):
    """Correct dynamic background in chunk of patterns by subtracting
    or dividing by a blurred version of each pattern.

    Returned pattern intensities are stretched to fill the input data
    type range.

    Parameters
    ----------
    patterns : dask.array.Array
        Patterns to correct dynamic background in.
    sigma : int, float or None
        Standard deviation of the gaussian kernel.
    operation : 'subtract' or 'divide', optional
        Subtract (default) or divide by dynamic background pattern.
    dtype_out : numpy.dtype, optional
        Data type of corrected patterns. If ``None`` (default), it is
        set to the same data type as the input patterns.

    Returns
    -------
    corrected_patterns : dask.array.Array
        Dynamic background corrected patterns.
    """

    if dtype_out is None:
        dtype_out = patterns.dtype

    corrected_patterns = np.empty_like(patterns, dtype=dtype_out)
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        pattern = patterns[nav_idx]
        blurred = gaussian_filter(pattern, sigma=sigma)
        if operation == "subtract":
            corrected_pattern = pattern - blurred
        else:  # Divide
            corrected_pattern = pattern / blurred
        corrected_patterns[nav_idx] = _rescale_pattern(
            corrected_pattern, dtype_out=dtype_out
        )

    return corrected_patterns


def _adaptive_histogram_equalization_chunk(
    patterns, kernel_size, clip_limit=0, nbins=128
):
    """Local contrast enhancement on chunk of patterns with adaptive
    histogram equalization.

    This method makes use of
    :func:`skimage.exposure.equalize_adapthist`.

    Parameters
    ----------
    patterns : dask.array.Array
        Patterns to enhance.
    kernel_size : int or list-like
        Shape of contextual regions for adaptive histogram equalization.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give
        more contrast). Default is 0.
    nbins : int, optional
        Number of gray bins for histogram ("data range"), default is
        128.

    Returns
    -------
    equalized_patterns : dask.array.Array
        Patterns with enhanced contrast.
    """

    dtype_in = patterns.dtype.type
    equalized_patterns = np.empty_like(patterns)
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        equalized_pattern = equalize_adapthist(
            patterns[nav_idx],
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins,
        )
        equalized_patterns[nav_idx] = _rescale_pattern(
            equalized_pattern, dtype_out=dtype_in
        )

    return equalized_patterns


def _average_neighbour_patterns_chunk(
    patterns, n_averaged, kernel, dtype_out=None
):
    """Average neighbour pattern intensities inplace within a
    square kernel, within a chunk.

    Parameters
    ----------
    patterns : dask.array.Array
        Patterns to average, with some overlap with surrounding chunks.
    n_averaged : dask.array.Array
        Number of neighbour patterns that will be averaged with
    kernel : numpy.ndarray
        Averaging kernel.
    dtype_out : numpy.dtype, optional
        Data type of corrected patterns. If ``None`` (default), it is
        set to the same data type as the input patterns.

    Returns
    -------
    averaged_patterns : dask.array.Array
        Averaged patterns.
    """

    if dtype_out is None:
        dtype_out = patterns.dtype

    # Convolve patterns with averaging kernel
    convolved_patterns = convolve(
        patterns.astype(np.float32), weights=kernel, mode="constant", cval=0,
    )

    # Divide convolved patterns by number of neighbours averaged with
    averaged_patterns = np.empty_like(convolved_patterns, dtype=dtype_out)
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        averaged_patterns[nav_idx] = (
            convolved_patterns[nav_idx] / n_averaged[nav_idx]
        ).astype(dtype_out)

    return averaged_patterns


def _pattern_kernel(axes, n_neighbours=1, exclude_kernel_corners=True):
    """Create a pattern kernel with a given number of neighbours,
    possibly excluding corner patterns.

    Parameters
    ----------
    axes : hyperspy.axes.AxesManager
        A HyperSpy signal axes manager containing navigation and signal
        dimensions and shapes.
    n_neighbours : int, optional
        Number of neighbours on each side to average with, default
        is 1.
    exclude_kernel_corners : bool, optional
        Whether to exclude corners in averaging kernel, default is
        True.

    Returns
    -------
    kernel : numpy.ndarray
        Pattern kernel with same shape as navigation shape in the input
        axes manager.
    """

    n_nav_dim = axes.navigation_dimension
    ny, nx = axes.navigation_shape

    kernel_size = 1 + n_neighbours * 2

    # Check if kernel size is bigger than scan
    if (kernel_size > ny) or (kernel_size > nx):
        largest_n_neighbours = (max([ny, nx]) - 1) // 2
        raise ValueError(
            f"Kernel size ({kernel_size} x {kernel_size}) is larger than "
            f"scan size ({ny} x {nx}). n_neighbours={largest_n_neighbours} "
            "is the largest possible value."
        )
    kernel = np.ones((kernel_size,) * n_nav_dim)

    # If desired, exclude corners in kernel
    if exclude_kernel_corners and n_nav_dim > 1:
        kernel_centre = (kernel_size // 2,) * n_nav_dim

        # Create an 'open' mesh-grid of the same size as the kernel
        y, x = np.ogrid[:kernel_size, :kernel_size]
        distance_to_centre = np.sqrt(
            (x - kernel_centre[0]) ** 2 + (y - kernel_centre[0]) ** 2
        )
        mask = distance_to_centre > kernel_centre[0]
        kernel[mask] = 0

    return kernel


def normalised_correlation_coefficient(pattern, template, zero_normalised=True):
    """Calculate the normalised or zero-normalised correlation
    coefficient between a pattern and a template following
    [Gonzalez2008]_.

    Parameters
    ----------
    pattern : numpy.ndarray or dask.array.Array
        Pattern to compare the template to.
    template : numpy.ndarray or dask.array.Array
        Template pattern.
    zero_normalised : bool, optional
        Subtract local mean value of intensities (default is ``True``).

    Returns
    -------
    coefficient : float
        Correlation coefficient in range [-1, 1] if zero normalised,
        otherwise [0, 1].

    References
    ----------
    .. [Gonzalez2008] Gonzalez, Rafael C, Woods, Richard E: Digital\
        Image Processing, 3rd edition, Pearson Education, 954, 2008.
    """

    pattern = pattern.astype(np.float32)
    template = template.astype(np.float32)
    if zero_normalised:
        pattern = pattern - pattern.mean()
        template = template - template.mean()
    coefficient = np.sum(pattern * template) / np.sqrt(
        np.sum(pattern ** 2) * np.sum(template ** 2)
    )

    return coefficient

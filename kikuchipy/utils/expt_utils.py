# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
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

import numpy as np
import dask.array as da
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.util.dtype import dtype_range


def _rescale_pattern(
        pattern, in_range=None, out_range=None, dtype_out=None):
    """Rescale pattern intensities to fill the data type range using an
    approach inspired by `skimage.exposure.rescale_intensity`.

    Parameters
    ----------
    pattern : da.Array
        Pattern to rescale.
    in_range, out_range : tuple of int or float, optional
        Min./max. intensity values of input and output pattern. If None,
        (default) `in_range` is set to pattern min./max. If None
        (default), `out_range` is set to `dtype_out` min./max
        according to `skimage.util.dtype.dtype_range`, with min. equal
        to zero.
    dtype_out : np.dtype
        Data type of rescaled pattern. If None (default), it is set to
        the same data type as the input pattern.

    Returns
    -------
    rescaled_pattern : da.Array
        Rescaled pattern.
    """

    if dtype_out is None:
        dtype_out = pattern.dtype.type
    if in_range is None:
        imin, imax = (pattern.min(), pattern.max())
    else:
        imin, imax = in_range
        pattern.clip(imin, imax)
    if out_range is None or out_range in dtype_range:
        omin = 0
        try:
            _, omax = dtype_range[dtype_out]
        except KeyError:
            raise KeyError("Could not set ouput intensity range, since data "
                           "type {} is not recognised. Use any of {}".format(
                dtype_out, dtype_range))
    else:
        omin, omax = out_range

    rescaled_pattern = (pattern - imin) / float(imax - imin)
    return (rescaled_pattern * (omax - omin) + omin).astype(dtype_out)


def _rescale_pattern_chunk(
        patterns, in_range=None, out_range=None, dtype_out=None):
    """Rescale patterns in chunk to fill the data type range using an
    approach inspired by `skimage.exposure.rescale_intensity`, keeping
    relative intensities or not.

    Parameters
    ----------
    patterns : da.Array
        Patterns to rescale.
    in_range, out_range : tuple of int or float, optional
        Min./max. intensity values of input and output pattern. If None,
        (default) `in_range` is set to pattern min./max. If None
        (default), `out_range` is set to `dtype_out` min./max
        according to `skimage.util.dtype_out.dtype_range`, with min. equal
        to zero.
    dtype_out : np.dtype
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
            patterns[nav_idx], in_range=in_range, out_range=out_range,
            dtype_out=dtype_out)
    return rescaled_patterns


def _static_background_correction_chunk(
        patterns, static_bg, operation='subtract', in_range=None,
        dtype_out=None):
    """Correct static background in patterns in chunk by subtracting or
    dividing by a static background pattern. Returned pattern
    intensities are rescaled keeping relative intensities or not and
    stretched to fill the input data type range.

    Parameters
    ----------
    patterns : da.Array
        Patterns to correct static background in.
    static_bg : np.ndarray or  da.Array, optional
        Static background pattern. If not passed we try to read it
        from the signal metadata.
    operation : 'subtract' or 'divide', optional
        Subtract (default) or divide by static background pattern.
    in_range : tuple of int or float, optional
        Min./max. intensity values of input and output patterns. If
        None, (default) `in_range` is set to pattern min./max, losing
        relative intensities between patterns.
    dtype_out : np.dtype
        Data type of corrected patterns. If None (default), it is set to
        the same data type as the input patterns.

    Returns
    -------
    corrected_patterns : da.Array
        Patterns with static background corrected.
    """

    dtype_in = patterns.dtype.type
    corrected_patterns = np.empty_like(patterns, dtype=dtype_in)
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        if operation == 'subtract':
            corrected_pattern = da.subtract(
                patterns[nav_idx], static_bg, dtype=dtype_in)
        else:  # Divide
            corrected_pattern = da.divide(
                patterns[nav_idx], static_bg, dtype=dtype_in)
        corrected_patterns[nav_idx] = _rescale_pattern(
            corrected_pattern, in_range=in_range, dtype_out=dtype_out)
    return corrected_patterns


def _adaptive_histogram_equalization_chunk(
        patterns, kernel_size, clip_limit, nbins):
    """Local contrast enhancement on chunk of patterns using adaptive
    histogram equalization as implemented in `scikit-image`.

    Parameters
    ----------
    patterns : da.Array
        Patterns to enhance.
    kernel_size : int or list-like, optional
        Shape of contextual regions for adaptive histogram equalization.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give
        more contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").

    Returns
    -------
    equalized_patterns : da.Array
        Chunk of enhanced patterns.
    """

    equalized_patterns = np.zeros_like(patterns)
    dtype_in = patterns.dtype.type
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        equalized_pattern = equalize_adapthist(
            patterns[nav_idx], kernel_size=kernel_size, clip_limit=clip_limit,
            nbins=nbins)
        equalized_patterns[nav_idx] = rescale_intensity(
            equalized_pattern, out_range=dtype_in)
    return equalized_patterns


def normalised_correlation_coefficient(
        pattern, template, zero_normalised=True):
    """Calculate the normalised or zero-normalised correlation
    coefficient between a pattern and a template following [1]_.

    Parameters
    ----------
    pattern : {np.ndarray, da.Array}
        Pattern to compare to template to.
    template : {np.ndarray, da.Array}
        Template pattern.
    zero_normalised : bool, optional
        Subtract local mean value of intensities.

    Returns
    -------
    coefficient : float
        Correlation coefficient in range [-1, 1] if zero normalised,
        otherwise [0, 1].

    References
    ----------
        .. [1] Gonzalez, Rafael C, Woods, Richard E: Digital Image
               Processing, 3rd edition, Pearson Education, 954, 2008.
    """

    pattern = pattern.astype(float)
    template = template.astype(float)
    if zero_normalised:
        pattern = pattern - pattern.mean()
        template = template - template.mean()
    coefficient = np.sum(pattern * template) / np.sqrt(np.sum(pattern ** 2) *
                                                       np.sum(template ** 2))
    return coefficient

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
from skimage.util import img_as_uint
from skimage.exposure import rescale_intensity
from skimage.exposure._adapthist import _clahe


def _rescale_pattern_chunk(patterns, imin, imax, dtype_out):
    """Rescale patterns in chunk to fill the data type range using
    `skimage.exposure.rescale_intensity`, keeping relative intensities
    or not.

    Parameters
    ----------
    patterns : da.Array
        Patterns to rescale.
    imin, imax : {None, int, float}
        Min./max. intensity values of input patterns.
    dtype_out : np.dtype
        Data type of output patterns.

    Returns
    -------
    rescaled_patterns : da.Array
        Rescaled patterns.
    """

    rescaled_patterns = np.zeros_like(patterns, dtype=dtype_out)
    in_range = (imin, imax)  # Scale relative to min./max. intensity in scan
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        if imin is None:  # Scale relative to min./max. intensity in pattern
            in_range = (patterns[nav_idx].min(), patterns[nav_idx].max())
        rescaled_patterns[nav_idx] = rescale_intensity(
            patterns[nav_idx], in_range=in_range, out_range=dtype_out)
    return rescaled_patterns


def _adaptive_histogram_equalization_pattern(
        pattern, kernel_size, clip_limit, nbins):
    """Local contrast enhancement of a pattern using adaptive histogram
    equalization as implemented in `scikit-image`.

    Parameters
    ----------
    pattern : da.Array
        Pattern to enhance.
    kernel_size : int or list-like, optional
        Shape of contextual regions for adaptive histogram equalization.
    nbins : int, optional
        Number of gray bins for histogram ("data range").
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give
        more contrast).

    Returns
    -------
    equalized_pattern : da.Array
        Enhanced pattern.
    """

    # Necessary preparations of pattern intensity for _clahe (range 2**14 is
    # hard-coded in scikit-image)
    dtype_in = pattern.dtype.type
    pattern = img_as_uint(pattern)
    pattern = rescale_intensity(pattern, out_range=(0, 2**14 - 1))

    # Perform adaptive histogram equalization
    equalized_pattern = _clahe(
        pattern, kernel_size=kernel_size, clip_limit=clip_limit * nbins,
        nbins=nbins)

    # Rescale intensity to fill input data type range
    equalized_pattern = rescale_intensity(equalized_pattern, out_range=dtype_in)

    return equalized_pattern


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
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        equalized_patterns[nav_idx] = _adaptive_histogram_equalization_pattern(
            patterns[nav_idx], kernel_size=kernel_size, clip_limit=clip_limit,
            nbins=nbins)
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
    coefficient = np.sum(pattern * template) / np.sqrt(np.sum(pattern**2) *
                                                       np.sum(template**2))
    return coefficient

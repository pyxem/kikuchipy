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
import kikuchipy as kp
import scipy.ndimage as scn
from skimage.exposure._adapthist import _clahe


def rescale_pattern_intensity(signal, out_range=np.uint8,
                              relative=False):
    """Scale pattern intensities in an EBSD signal inplace.

    Relative intensities between patterns can be maintained by passing
    `relative=True`. The desired `output_range` can be specified by a
    tuple of (min., max.) intensity or a data type.

    Parameters
    ----------
    signal : kp.signals.EBSD or kp.lazy_signals.LazyEBSD
        Signal instance with four-dimensional data.
    out_range : dtype or tuple, optional
        Output intensity range. If a tuple is passed, the output data
        type is np.float32.
    relative : bool, optional
        Keep relative intensities between patterns.

    Returns
    -------
    rescaled_patterns : np.array or da.Array
        Rescaled EBSD patterns.
    """

    # Get valid intensity range and data type for rescaled patterns
    omin, omax = intensity_range(out_range)
    if not isinstance(out_range, tuple) and (
            np.issubdtype(out_range, np.integer) or
            np.issubdtype(out_range, np.float)):
        dtype = out_range
    else:
        dtype = np.float32

    patterns = signal.data
    if relative:  # Get min. and max. intensity in scan
        imin = patterns.min()
        imax = patterns.max()
        if isinstance(patterns, da.Array):
            imin = imin.compute(show_progressbar=False)
            imax = imax.compute(show_progressbar=False)
        scale = omax / imax
        rescaled_patterns = ((patterns-imin)*scale + omin).astype(dtype)
    else:  # Get min. and max. intensity per pattern
        signal_axes = signal.axes_manager.signal_axes
        imin = signal.min(signal_axes).data[:, :, np.newaxis, np.newaxis]
        imax = signal.max(signal_axes).data[:, :, np.newaxis, np.newaxis]
        rescaled_patterns = (patterns-imin) / (imax-imin)
        rescaled_patterns = (rescaled_patterns*(omax-omin) + omin).astype(dtype)

    return rescaled_patterns


def intensity_range(in_range):
    """Return intensity range (min, max) based on desired value type.

    Parameters
    ----------
    in_range : dtype or tuple
        Instance to return (min, max) from.

    Returns
    -------
    imin, imax : int or float
        Intensity range of `range`.
    """

    if isinstance(in_range, tuple):
        imin, imax = in_range
    elif np.issubdtype(in_range, np.integer):
        imin, imax = np.iinfo(in_range).min, np.iinfo(in_range).max
    elif np.issubdtype(in_range, np.float):
        imin, imax = np.finfo(in_range).min, np.finfo(in_range).max
    else:
        raise ValueError("{} is not a valid in_range".format(in_range))

    return imin, imax


def static_correction(pattern, operation, static_bg, imin, scale):
    """Correct static background using a static background pattern.

    Parameters
    ----------
    pattern : {np.ndarray, da.Array}
        Signal pattern.
    operation : {'divide', 'static'}
        Divide or subtract by static background pattern.
    static_bg : {np.ndarray, da.Array}
        Static background pattern.
    imin : int
        Minimum intensity of input pattern.
    scale : int, float
        Scaling factor for intensities of output pattern.

    Returns
    -------
    corrected_pattern : {np.ndarray, da.Array}
        Static background corrected pattern.
    """
    pattern = pattern.astype(float)

    if operation == 'divide':
        corrected_pattern = pattern / static_bg
    else:
        corrected_pattern = pattern - static_bg

    return rescale_pattern_intensity(corrected_pattern, imin=imin, scale=scale)


def dynamic_correction(pattern, operation, sigma):
    """Correct dynamic background using a gaussian blurred version of
    the pattern.

    Parameters
    ----------
    pattern : {np.ndarray, da.Array}
        Signal pattern.
    operation : {'divide', 'subtract'}
        Divide or subtract by dynamic pattern.
    sigma : {int, float}
        Standard deviation of the gaussian kernel. If None
        (default), a deviation of pattern width/30 is chosen.

    Returns
    -------
    corrected_pattern : {np.ndarray, da.Array}
        Dynamic background corrected pattern.
    """
    pattern = pattern.astype(float)
    blurred_pattern = scn.gaussian_filter(pattern, sigma, truncate=2.0)

    if operation == 'divide':
        corrected_pattern = pattern / blurred_pattern
    else:
        corrected_pattern = pattern - blurred_pattern

    return rescale_pattern_intensity(corrected_pattern)


def equalize_adapthist_pattern(pattern, kernel_size, clip_limit=0.01,
                               nbins=256):
    """Local contrast enhancement of an electron backscatter diffraction
    pattern using contrast limited adaptive histogram equalisation
    (CLAHE).

    Parameters
    ----------
    pattern : array_like
        Two-dimensional array containing signal.
    kernel_size : integer or list-like
        Defines the shape of contextual regions used in the algorithm.
    clip_limit : float, optional
        Clipping limit, normalised between 0 and 1 (higher values give
        more contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").

    Returns
    -------
    pattern : array_like
        Equalised pattern.

    Notes
    -----
    Adapted from scikit-image, returning the pattern with correct data
    type. See ``skimage.exposure.equalize_adapthist`` documentation for
    more details.
    """
    # Rescale pattern to 16-bit [0, 2**14 - 1]
    pattern = rescale_pattern_intensity(pattern, omax=2**14 - 1,
                                        dtype_out=np.uint16)

    # Perform CLAHE and rescale to 8-bit [0, 255]
    pattern = _clahe(pattern, kernel_size, clip_limit * nbins, nbins)
    pattern = rescale_pattern_intensity(pattern)

    return pattern


def normalised_correlation_coefficient(pattern, template,
                                       zero_normalised=True):
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

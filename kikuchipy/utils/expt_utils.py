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
from scipy.ndimage import gaussian_filter, median_filter
from hyperspy.api import plot
from skimage.exposure._adapthist import _clahe


def rescale_pattern_intensity(pattern, imin=None, scale=None,
                              dtype_out=np.uint8):
    """Rescale electron backscatter diffraction pattern intensities to
    unsigned integer range or desired unsigned range specified by imin
    and scale. If imin and scale are passed the pattern intensities are
    stretched to a global min. and max. intensity according to these
    values. Otherwise they are stretched to between zero and maximum of
    dtype_out.

    Parameters
    ----------
    pattern : array_like
        Two-dimensional array containing signal.
    imin : int, optional
        Global min. intensity of patterns.
    scale : float, optional
        Global scaling factor for intensities of output pattern.
    dtype_out : numpy dtype
        Data type of output pattern.

    Returns
    -------
    pattern : array_like
        Output pattern rescaled to specified range.
    """
    # TODO: Stop function from leaking memory when used with map
    if np.issubdtype(dtype_out, np.unsignedinteger) is False:
        raise ValueError("Data type is not unsigned integer.")

    if imin is None and scale is None:  # Local contrast stretching
        omax = np.iinfo(dtype_out).max
        imin = pattern.min()
        scale = float(omax / (pattern.max() - imin))

    # Set lowest intensity to zero and scale intensities
    pattern = (pattern - imin) * scale

    return pattern.astype(dtype_out)


def correct_background(pattern, static, dynamic, bg, sigma, imin, scale):
    """Static and dynamic background correction on an electron
    backscatter diffraction pattern.

    Parameters
    ----------
    pattern : array_like
        Two-dimensional array containing signal.
    static : bool, optional
        If True, static correction is performed.
    dynamic : bool, optional
        If True, dynamic correction is performed.
    bg : array_like
        Background image for static correction.
    sigma : int, float
        Standard deviation for the gaussian kernel for dynamic
        correction.
    imin : int
        Global min. intensity of patterns.
    scale : int, float
        Global scaling factor for intensities of output pattern.

    Returns
    -------
    pattern : array_like
        Output pattern with background corrected and intensities
        stretched to a desired range.
    """
    if static:
        # Change data types to avoid negative intensities in subtraction
        dtype = np.int16
        pattern = pattern.astype(dtype)
        bg = bg.astype(dtype)

        # Subtract static background
        pattern = pattern - bg

        # Rescale intensities, either keeping relative intensities or not
        pattern = rescale_pattern_intensity(pattern, imin=imin, scale=scale)

    if dynamic:
        # Create gaussian blurred version of pattern
        blurred = gaussian_filter(pattern, sigma, truncate=2.0)

        # Change data types to avoid negative intensities in subtraction
        dtype = np.int16
        pattern = pattern.astype(dtype)
        blurred = blurred.astype(dtype)

        # Subtract blurred background
        pattern = pattern - blurred

        # Rescale intensities, loosing relative intensities
        pattern = rescale_pattern_intensity(pattern)

    return pattern


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


def remove_dead(pattern, deadpixels, deadvalue='average', d=1):
    """Remove dead pixels from a pattern.

    Parameters
    ----------
    pattern : array_like
        Two-dimensional array containing signal.
    deadpixels : array_like
        Array containing the array indices of dead pixels in the
        pattern.
    deadvalue : string
        Specify how deadpixels should be treated, options are;
            'average': takes the average of adjacent pixels
            'nan':  sets the dead pixel to nan
    d : int, optional
        Number of adjacent pixels to average over.

    Returns
    -------
    new_pattern : array_like
        Two-dimensional array containing z with dead pixels removed.

    Notes
    -----
    This function is slow for lazy signals and leaks memory.
    """
    # TODO: Stop function from leaking memory when used with map
    new_pattern = np.copy(pattern)
    if deadvalue == 'average':
        for (i, j) in deadpixels:
            neighbours = pattern[i - d:i + d + 1, j - d:j + d + 1].flatten()
            neighbours = np.delete(neighbours, 4)  # Exclude dead pixel
            new_pattern[i, j] = int(np.mean(neighbours))
    elif deadvalue == 'nan':
        for (i, j) in deadpixels:
            new_pattern[i, j] = np.nan
    else:
        raise NotImplementedError("The method specified is not implemented. "
                                  "See documentation for available "
                                  "implementations.")
    return new_pattern


def find_deadpixels_single_pattern(pattern, threshold=5, to_plot=False,
                                   mask=None):
    """Find dead pixels in one experimentally acquired diffraction
    patterns by comparing pixel values in a blurred version of the
    selected pattern to the original pattern. If the intensity
    difference is above a threshold the pixel is labeled as dead.

    Parameters
    ----------
    pattern : array_like
        Two-dimensional array containing signal.
    threshold : int, optional
        Threshold for difference in pixel intensities between
        blurred and original pattern. The actual threshold is given
        as threshold*(standard deviation of the difference between
        blurred and original pattern).
    to_plot : bool, optional
        If True (default is False), the pattern with the dead pixels
        highlighted is plotted.
    mask : array of bool, optional
        No deadpixels are found where mask is True. The shape of pattern
        and mask must be the same.

    Returns
    -------
    deadpixels : list of tuples
        List of tuples containing pattern indices for dead pixels.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from kikuchipy.utils.expt_utils import \
            find_deadpixels_single_pattern
        # Threshold the first pattern, so that pixels with an intensity
        # below 60 will be masked.
        pattern = s.inav[0, 0].data
        mask = np.zeros(s.axes_manager.signal_shape)
        mask[np.where(pattern < 60)] = True
        deadpixels = find_deadpixels_single_pattern(pattern, mask=mask)
    """
    if isinstance(pattern, da.Array):
        pattern = pattern.compute(show_progressbar=False)

    pattern = pattern.astype(np.int16)
    blurred = median_filter(pattern, size=2)
    difference = pattern - blurred
    threshold = threshold * np.std(difference)

    # Find the dead pixels (ignoring border pixels)
    deadpixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
    deadpixels = np.array(deadpixels) + 1
    deadpixels = list(map(tuple, deadpixels.T))  # List of tuples

    # If a mask is given, check if any of the deadpixels are also found
    # within the mask, and if so, delete those.
    if mask is not None:
        deadpixels = np.array(deadpixels)
        mask_indices = np.concatenate(([np.where(mask)[0]],
                                       [np.where(mask)[1]]), axis=0).T
        for deadpixel in deadpixels:
            common_indices = np.intersect1d(
                np.where(mask_indices[:, 0] == deadpixel[0]),
                np.where(mask_indices[:, 1] == deadpixel[1]))
            if common_indices.size:
                where_indices = np.where(np.array(
                    deadpixels == mask_indices[common_indices]))
                w_ind_x, w_ind_y = where_indices[:][0], where_indices[:][1]
                delete_indices = np.array([], dtype=np.int16)
                for n in range(1, len(w_ind_x)):
                    if (w_ind_x[n - 1] == w_ind_x[n] and w_ind_y[n - 1] == 0
                            and w_ind_y[n] == 1):
                        delete_indices = np.append(delete_indices, w_ind_x[n])
                deadpixels = np.delete(deadpixels, delete_indices, axis=0)
        deadpixels = tuple(map(tuple, deadpixels))

    if to_plot:
        plot_markers_single_pattern(pattern, deadpixels)

    return deadpixels


def plot_markers_single_pattern(pattern, markers):
    """Plot markers on an electron backscatter diffraction pattern.

    Parameters
    ----------
    pattern : array_like
    markers : list of tuples
    """
    pat = kp.signals.EBSD(pattern)
    pat.plot()
    for (y, x) in markers:
        m = plot.markers.point(x, y, color='red')
        pat.add_marker(m, permanent=False)

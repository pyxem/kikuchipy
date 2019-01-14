# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import gaussian_filter


def rescale_pattern_intensity(pattern, imin=None, scale=None, omax=255,
                              dtype_out=np.uint8):
    """Rescale electron backscatter diffraction pattern intensities to
    specified range using contrast stretching.

    If imin and scale is passed the pattern intensities are stretched to a
    global min. and max. intensity. Otherwise they are stretched to between
    zero and omax.

    Parameters
    ----------
    pattern : numpy array of unsigned integer dtype
        Input pattern.
    imin : int, optional
        Global min. intensity of patterns.
    scale : float, optional
        Global scaling factor for intensities of output pattern.
    omax : int, optional
        Max. intensity of output pattern (default = 255).
    dtype_out : numpy dtype
        Data type of output pattern.

    Returns
    -------
    pattern : numpy array
        Output pattern rescaled to specified range.
    """
    if imin is None and scale is None:  # Local contrast stretching
        imin = pattern.min()
        scale = float(omax / (pattern.max() + abs(imin)))

    # Set lowest intensity to zero
    pattern = pattern + abs(imin)

    # Scale intensities
    return np.array(pattern * scale, dtype=dtype_out)


def correct_background(pattern, static, dynamic, bg, sigma, imin, scale):
    """Perform background correction on an electron backscatter diffraction
    patterns.

    Parameters
    ----------
    pattern : numpy array of unsigned integer dtype
        Input pattern.
    static : bool, optional
        If True, static correction is performed.
    dynamic : bool, optional
        If True, dynamic correction is performed.
    bg : numpy array
        Background image for static correction.
    sigma : int, float
        Standard deviation for the gaussian kernel for dynamic correction.
    imin : int
        Global min. intensity of patterns.
    scale : int, float
        Global scaling factor for intensities of output pattern.

    Returns
    -------
    pattern : numpy array
        Output pattern with background corrected and intensities stretched to
        a desired range.
    """
    if static:
        # Change data types to avoid negative intensities in subtraction
        dtype = np.int8
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

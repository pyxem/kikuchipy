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

from typing import Union, Tuple, Optional, List

from numba import njit
import numpy as np
from numpy.fft import (
    fft2,
    rfft2,
    ifft2,
    irfft2,
    fftshift,
    ifftshift,
)
from scipy.ndimage import gaussian_filter
from skimage.util.dtype import dtype_range

from kikuchipy.filters.fft_barnes import _fft_filter, _fft_filter_setup
from kikuchipy.filters.window import Window


def rescale_intensity(
    pattern: np.ndarray,
    in_range: Optional[Tuple[Union[int, float], ...]] = None,
    out_range: Optional[Tuple[Union[int, float], ...]] = None,
    dtype_out: Optional[np.dtype] = None,
) -> np.ndarray:
    """Rescale intensities in an EBSD pattern.

    Pattern max./min. intensity is determined from `out_range` or the
    data type range of :class:`numpy.dtype` passed to `dtype_out`.

    This method is based on :func:`skimage.exposure.rescale_intensity`.

    Parameters
    ----------
    pattern
        EBSD pattern.
    in_range
        Min./max. intensity values of the input pattern. If None
        (default), it is set to the pattern's min./max intensity.
    out_range
        Min./max. intensity values of the rescaled pattern. If None
        (default), it is set to `dtype_out` min./max according to
        `skimage.util.dtype.dtype_range`.
    dtype_out
        Data type of the rescaled pattern. If None (default), it is set
        to the same data type as the input pattern.

    Returns
    -------
    rescaled_pattern : numpy.ndarray
        Rescaled pattern.
    """
    if dtype_out is None:
        dtype_out = pattern.dtype.type

    if in_range is None:
        imin, imax = np.min(pattern), np.max(pattern)
    else:
        imin, imax = in_range
        pattern = np.clip(pattern, imin, imax)

    if out_range is None or out_range in dtype_range:
        try:
            if isinstance(dtype_out, np.dtype):
                dtype_out = dtype_out.type
            omin, omax = dtype_range[dtype_out]
        except KeyError:
            raise KeyError(
                "Could not set output intensity range, since data type "
                f"'{dtype_out}' is not recognised. Use any of '{dtype_range}'."
            )
    else:
        omin, omax = out_range

    return _rescale(pattern, imin, imax, omin, omax).astype(dtype_out)


@njit
def _rescale(
    pattern: np.ndarray,
    imin: Union[int, float],
    imax: Union[int, float],
    omin: Union[int, float],
    omax: Union[int, float],
) -> np.ndarray:
    rescaled_pattern = (pattern - imin) / float(imax - imin)
    return rescaled_pattern * (omax - omin) + omin


def remove_dynamic_background(
    pattern: np.ndarray,
    operation: str = "subtract",
    filter_domain: str = "frequency",
    std: Union[None, int, float] = None,
    truncate: Union[int, float] = 4.0,
    dtype_out: Union[
        None, np.dtype, Tuple[int, int], Tuple[float, float]
    ] = None,
) -> np.ndarray:
    """Remove the dynamic background in an EBSD pattern.

    The removal is performed by subtracting or dividing by a Gaussian
    blurred version of the pattern. The blurred version is obtained
    either in the frequency domain, by a low pass Fast Fourier Transform
    (FFT) Gaussian filter, or in the spatial domain by a Gaussian
    filter. Returned pattern intensities are rescaled to fill the input
    data type range.

    Parameters
    ----------
    pattern
        EBSD pattern.
    operation
        Whether to "subtract" (default) or "divide" by the dynamic
        background pattern.
    filter_domain
        Whether to obtain the dynamic background by applying a Gaussian
        convolution filter in the "frequency" (default) or "spatial"
        domain.
    std
        Standard deviation of the Gaussian window. If None (default), it
        is set to width/8.
    truncate
        Truncate the Gaussian window at this many standard deviations.
        Default is 4.0.
    dtype_out
        Data type of corrected pattern. If None (default), it is set to
        input patterns' data type.

    Returns
    -------
    corrected_pattern : numpy.ndarray
        Pattern with the dynamic background removed.

    See Also
    --------
    kikuchipy.signals.EBSD.remove_dynamic_background
    kikuchipy.pattern.remove_dynamic_background
    """
    if std is None:
        std = pattern.shape[1] / 8

    if dtype_out is None:
        dtype_out = pattern.dtype.type

    if filter_domain == "frequency":
        (
            fft_shape,
            kernel_shape,
            kernel_fft,
            offset_before_fft,
            offset_after_ifft,
        ) = _dynamic_background_frequency_space_setup(
            pattern_shape=pattern.shape, std=std, truncate=truncate,
        )
        dynamic_bg = _fft_filter(
            image=pattern,
            fft_shape=fft_shape,
            window_shape=kernel_shape,
            transfer_function=kernel_fft,
            offset_before_fft=offset_before_fft,
            offset_after_ifft=offset_after_ifft,
        )
    elif filter_domain == "spatial":
        dynamic_bg = gaussian_filter(
            input=pattern, sigma=std, truncate=truncate,
        )
    else:
        filter_domains = ["frequency", "spatial"]
        raise ValueError(f"{filter_domain} must be either of {filter_domains}.")

    # Remove dynamic background
    if operation == "subtract":
        corrected_pattern = pattern - dynamic_bg
    else:  # operation == "divide"
        corrected_pattern = pattern / dynamic_bg

    # Rescale intensity
    corrected_pattern = rescale_intensity(
        corrected_pattern, dtype_out=dtype_out
    )

    return corrected_pattern


def _dynamic_background_frequency_space_setup(
    pattern_shape: Union[List[int], Tuple[int, int]],
    std: Union[int, float],
    truncate: Union[int, float],
) -> Tuple[
    Tuple[int, int],
    Tuple[int, int],
    np.ndarray,
    Tuple[int, int],
    Tuple[int, int],
]:
    # Get Gaussian filtering window
    shape = (int(truncate * std),) * 2
    window = Window("gaussian", std=std, shape=shape)
    window = window / (2 * np.pi * std ** 2)
    window = window / np.sum(window)

    # FFT filter setup
    (
        fft_shape,
        transfer_function,
        offset_before_fft,
        offset_after_ifft,
    ) = _fft_filter_setup(pattern_shape, window)

    return (
        fft_shape,
        window.shape,
        transfer_function,
        offset_before_fft,
        offset_after_ifft,
    )


def get_dynamic_background(
    pattern: np.ndarray,
    filter_domain: str = "frequency",
    std: Union[None, int, float] = None,
    truncate: Union[int, float] = 4.0,
) -> np.ndarray:
    """Get the dynamic background in an EBSD pattern.

    The background is obtained either in the frequency domain, by a low
    pass Fast Fourier Transform (FFT) Gaussian filter, or in the spatial
    domain by a Gaussian filter.

    Data type is preserved.

    Parameters
    ----------
    pattern
        EBSD pattern.
    filter_domain
        Whether to obtain the dynamic background by applying a Gaussian
        convolution filter in the "frequency" (default) or "spatial"
        domain.
    std
        Standard deviation of the Gaussian window. If None (default), a
        deviation of pattern width/8 is chosen.
    truncate
        Truncate the Gaussian window at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    dynamic_bg : numpy.ndarray
        The dynamic background.
    """
    if std is None:
        std = pattern.shape[1] / 8

    if filter_domain == "frequency":
        (
            fft_shape,
            kernel_shape,
            kernel_fft,
            offset_before_fft,
            offset_after_ifft,
        ) = _dynamic_background_frequency_space_setup(
            pattern_shape=pattern.shape, std=std, truncate=truncate,
        )
        dynamic_bg = _fft_filter(
            image=pattern,
            fft_shape=fft_shape,
            window_shape=kernel_shape,
            transfer_function=kernel_fft,
            offset_before_fft=offset_before_fft,
            offset_after_ifft=offset_after_ifft,
        )
    elif filter_domain == "spatial":
        dynamic_bg = gaussian_filter(
            input=pattern, sigma=std, truncate=truncate,
        )
    else:
        filter_domains = ["frequency", "spatial"]
        raise ValueError(f"{filter_domain} must be either of {filter_domains}.")

    return dynamic_bg.astype(pattern.dtype)


def get_image_quality(
    pattern: np.ndarray,
    normalize: bool = True,
    frequency_vectors: Optional[np.ndarray] = None,
    inertia_max: Union[None, int, float] = None,
) -> float:
    """Return the image quality of an EBSD pattern.

    The image quality is calculated based on the procedure defined by
    Krieger Lassen [Lassen1994]_.

    Parameters
    ----------
    pattern
        EBSD pattern.
    normalize
        Whether to normalize the pattern to a mean of zero and standard
        deviation of 1 before calculating the image quality (default is
        True).
    frequency_vectors
        Integer 2D array assigning each FFT spectrum frequency component
        a weight. If None (default), these are calculated from
        :func:`~kikuchipy.pattern.fft_frequency_vectors`. This only
        depends on the pattern shape.
    inertia_max
        Maximum possible inertia of the FFT power spectrum of the image.
        If None (default), this is calculated from the
        `frequency_vectors`, which in this case *must* be passed. This
        only depends on the pattern shape.

    Returns
    -------
    image_quality : numpy.ndarray
        Image quality of the pattern.
    """
    if frequency_vectors is None:
        sy, sx = pattern.shape
        frequency_vectors = fft_frequency_vectors((sy, sx))

    if inertia_max is None:
        sy, sx = pattern.shape
        inertia_max = np.sum(frequency_vectors) / (sy * sx)

    if normalize is True:
        pattern = normalize_intensity(pattern)

    # Compute FFT
    # TODO: Reduce frequency vectors to real part only to enable real part FFT
    fft_pattern = fft2(pattern)

    # Obtain (un-shifted) FFT spectrum
    spectrum = fft_spectrum(fft_pattern)

    # Calculate inertia (see Lassen1994)
    inertia = np.sum(spectrum * frequency_vectors) / np.sum(spectrum)

    return 1 - (inertia / inertia_max)


def fft(
    pattern: np.ndarray,
    apodization_window: Union[None, np.ndarray, Window] = None,
    shift: bool = False,
    real_fft_only: bool = False,
    **kwargs,
) -> np.ndarray:
    """Compute the discrete Fast Fourier Transform (FFT) of an EBSD
    pattern.

    Very light wrapper around routines in :mod:`scipy.fft`. The routines
    are wrapped instead of used directly to accommodate easy setting of
    `shift` and `real_fft_only`.

    Parameters
    ----------
    pattern
        EBSD pattern.
    apodization_window
        An apodization window to apply before the FFT in order to
        suppress streaks.
    shift
        Whether to shift the zero-frequency component to the centre of
        the spectrum (default is False).
    real_fft_only
        If True, the discrete FFT is computed for real input using
        :func:`scipy.fft.rfft2`. If False (default), it is computed
        using :func:`scipy.fft.fft2`.
    kwargs :
        Keyword arguments pass to :func:`scipy.fft.fft2` or
        :func:`scipy.fft.rfft2`.

    Returns
    -------
    out : numpy.ndarray
        The result of the 2D FFT.
    """
    if apodization_window is not None:
        pattern = pattern * apodization_window

    if real_fft_only:
        fft_use = rfft2
    else:
        fft_use = fft2

    if shift:
        out = fftshift(fft_use(pattern, **kwargs))
    else:
        out = fft_use(pattern, **kwargs)

    return out


def ifft(
    fft_pattern: np.ndarray,
    shift: bool = False,
    real_fft_only: bool = False,
    **kwargs,
) -> np.ndarray:
    """Compute the inverse Fast Fourier Transform (IFFT) of an FFT of an
    EBSD pattern.

    Very light wrapper around routines in :mod:`scipy.fft`. The routines
    are wrapped instead of used directly to accommodate easy setting of
    `shift` and `real_fft_only`.

    Parameters
    ----------
    fft_pattern
        FFT of EBSD pattern.
    shift
        Whether to shift the zero-frequency component back to the
        corners of the spectrum (default is False).
    real_fft_only
        If True, the discrete IFFT is computed for real input using
        :func:`scipy.fft.irfft2`. If False (default), it is computed
        using :func:`scipy.fft.ifft2`.
    kwargs :
        Keyword arguments pass to :func:`scipy.fft.ifft`.

    Returns
    -------
    pattern : numpy.ndarray
        Real part of the IFFT of the EBSD pattern.
    """
    if real_fft_only:
        fft_use = irfft2
    else:
        fft_use = ifft2

    if shift:
        pattern = fft_use(ifftshift(fft_pattern, **kwargs))
    else:
        pattern = fft_use(fft_pattern, **kwargs)

    return np.real(pattern)


def fft_filter(
    pattern: np.ndarray,
    transfer_function: Union[np.ndarray, Window],
    apodization_window: Union[None, np.ndarray, Window] = None,
    shift: bool = False,
) -> np.ndarray:
    """Filter an EBSD patterns in the frequency domain.

    Parameters
    ----------
    pattern
        EBSD pattern.
    transfer_function
        Filter transfer function in the frequency domain.
    apodization_window
        An apodization window to apply before the FFT in order to
        suppress streaks.
    shift
        Whether to shift the zero-frequency component to the centre of
        the spectrum. Default is False.

    Returns
    -------
    filtered_pattern : numpy.ndarray
        Filtered EBSD pattern.
    """
    # Get the FFT
    pattern_fft = fft(
        pattern, shift=shift, apodization_window=apodization_window
    )

    # Apply the transfer function to the FFT
    filtered_fft = pattern_fft * transfer_function

    # Get real part of IFFT of the filtered FFT
    return np.real(ifft(filtered_fft, shift=shift))


@njit
def fft_spectrum(fft_pattern: np.ndarray) -> np.ndarray:
    """Compute the FFT spectrum of a Fourier transformed EBSD pattern.

    Parameters
    ----------
    fft_pattern
        Fourier transformed EBSD pattern.

    Returns
    -------
    fft_spectrum : numpy.ndarray
        2D FFT spectrum of the EBSD pattern.
    """
    return np.sqrt(fft_pattern.real ** 2 + fft_pattern.imag ** 2)


@njit
def normalize_intensity(
    pattern: np.ndarray, num_std: int = 1, divide_by_square_root: bool = False
) -> np.ndarray:
    """Normalize image intensities to a mean of zero and a given
    standard deviation.

    Data type is preserved.

    Parameters
    ----------
    pattern
        EBSD pattern.
    num_std
        Number of standard deviations of the output intensities (default
        is 1).
    divide_by_square_root
        Whether to divide output intensities by the square root of the
        image size (default is False).

    Returns
    -------
    normalized_pattern : numpy.ndarray
        Normalized pattern.

    Notes
    -----
    Data type should always be changed to floating point, e.g.
    ``np.float32`` with :meth:`numpy.ndarray.astype`, before normalizing
    the intensities.
    """
    pattern_mean = np.mean(pattern)
    pattern_std = np.std(pattern)

    if divide_by_square_root:
        return (pattern - pattern_mean) / (
            num_std * pattern_std * np.sqrt(pattern.size)
        )
    else:
        return (pattern - pattern_mean) / (num_std * pattern_std)


def fft_frequency_vectors(shape: Tuple[int, int]) -> np.ndarray:
    """Get the frequency vectors in a Fourier Transform spectrum.

    Parameters
    ----------
    shape
        Fourier transform shape.

    Returns
    -------
    frequency_vectors : numpy.ndarray
        Frequency vectors.
    """
    sy, sx = shape

    linex = np.arange(sx) + 1
    linex[sx // 2 :] -= sx + 1
    liney = np.arange(sy) + 1
    liney[sy // 2 :] -= sy + 1

    frequency_vectors = np.empty(shape=(sy, sx))
    for i in range(sy):
        frequency_vectors[i] = liney[i] ** 2 + linex ** 2 - 1

    return frequency_vectors


def _zero_mean(patterns: np.ndarray, axis: Tuple[int, tuple]) -> np.ndarray:
    patterns_mean = patterns.mean(axis=axis, keepdims=True)
    return patterns - patterns_mean


def _normalize(patterns: np.ndarray, axis: Tuple[int, tuple]) -> np.ndarray:
    patterns_squared = patterns ** 2
    patterns_norm = patterns_squared.sum(axis=axis, keepdims=True)
    patterns_norm_squared = patterns_norm ** 0.5
    return patterns / patterns_norm_squared

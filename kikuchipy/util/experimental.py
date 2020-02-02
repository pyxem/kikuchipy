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
import warnings

import dask.array as da
import numpy as np
import numbers
from hyperspy.axes import AxesManager
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal.windows import get_window
from skimage.exposure import equalize_adapthist
from skimage.util.dtype import dtype_range


def _rescale_pattern(pattern, in_range=None, out_range=None, dtype_out=None):
    """Rescale pattern intensities inplace to desired
    :class:`numpy.dtype` range specified by ``dtype_out`` keeping
    relative intensities or not.

    This method is based upon
    :func:`skimage.exposure.rescale_intensity`.

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
    patterns, kernel_sums, kernel, dtype_out=None
):
    """Average neighbour pattern intensities within a kernel, within a
    chunk.

    Parameters
    ----------
    patterns : dask.array.Array
        Patterns to average, with some overlap with surrounding chunks.
    kernel_sums : dask.array.Array
        Sum of kernel coefficients for each pattern.
    kernel : numpy.ndarray
        Averaging kernel.
    dtype_out : numpy.dtype, optional
        Data type of averaged patterns. If ``None`` (default), it is
        set to the same data type as the input patterns.

    Returns
    -------
    averaged_patterns : dask.array.Array
        Averaged patterns.

    """

    if dtype_out is None:
        dtype_out = patterns.dtype

    # Convolve patterns with kernel
    convolved_patterns = convolve(
        patterns.astype(np.float32), weights=kernel, mode="constant", cval=0,
    )

    # Divide convolved patterns by number of neighbours averaged with
    averaged_patterns = np.empty_like(convolved_patterns, dtype=dtype_out)
    for nav_idx in np.ndindex(patterns.shape[:-2]):
        averaged_patterns[nav_idx] = (
            convolved_patterns[nav_idx] / kernel_sums[nav_idx]
        ).astype(dtype_out)

    return averaged_patterns


def get_pattern_kernel(
    kernel="circular", kernel_size=(3, 3), axes=None, **kwargs
):
    """Return a pattern kernel of a given shape with specified
    coefficients.

    See :func:`scipy.signal.windows.get_window` for available kernels
    and required arguments for that specific kernel.

    Parameters
    ----------
    kernel : 'circular', 'rectangular', 'gaussian', str, or
            :class:`numpy.ndarray`, optional
        Averaging kernel. Available kernel types are listed in
        :func:`scipy.signal.windows.get_window`, in addition to a
        circular kernel (default) filled with ones in which corners are
        excluded from averaging. A pattern is considered to be in a
        corner if its radial distance to the origin is shorter or equal
        to the kernel half width. A 1D or 2D numpy array with kernel
        coefficients can also be passed.
    kernel_size : int or tuple of ints, optional
        Size of averaging kernel if not a custom kernel is passed to
        `kernel`. This can be either 1D or 2D, and does not have to be
        symmetrical. Default is (3, 3).
    axes : None or hyperspy.axes.AxesManager, optional
        A HyperSpy signal axes manager containing navigation and signal
        dimensions and shapes can be passed to ensure that the averaging
        kernel is compatible with the signal.
    **kwargs :
        Keyword arguments passed to the available kernel type listed in
        :func:`scipy.signal.windows.get_window`.

    Returns
    -------
    returned_kernel : numpy.ndarray
        The pattern kernel of given shape with specified coefficients.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> kp.util.experimental.get_pattern_kernel(
            kernel="circular", kernel_size=(3, 3))
    array([[0., 1., 0.],
           [1., 1., 1.],
           [0., 1., 0.]])
    >>> kp.util.experimental.get_pattern_kernel(kernel="gaussian", std=2)
    array([[0.77880078, 0.8824969 , 0.77880078],
           [0.8824969 , 1.        , 0.8824969 ],
           [0.77880078, 0.8824969 , 0.77880078]])

    See Also
    --------
    scipy.signal.windows.get_window

    """

    # Overwrite towards the end if no custom kernel is passed
    returned_kernel = kernel

    # Get kernel size if a custom kernel is passed, at the same time checking
    # if the custom kernel's shape is valid
    if not isinstance(kernel, str):
        try:
            kernel_size = kernel.shape
        except AttributeError:
            raise ValueError(
                "Kernel must be of type numpy.ndarray, however a kernel of type"
                f" {type(kernel)} was passed."
            )

    # Make kernel_size a tuple if an integer was passed
    if isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,)

    # Kernel dimensions must be positive
    try:
        if any(np.array(kernel_size) < 0):
            raise ValueError(
                f"Kernel dimensions must be positive, however {kernel_size} was"
                " passed."
            )
    except TypeError:
        raise TypeError(
            "Kernel dimensions must be an int or a tuple of ints, however "
            f"kernel dimensions of type {type(kernel_size)} was passed."
        )

    if axes is not None:
        try:
            nav_shape = axes.navigation_shape
        except AttributeError:
            raise AttributeError(
                "A hyperspy.axes.AxesManager object must be passed to the "
                f"'axes' parameter, however a {type(axes)} was passed."
            )

        # Number of kernel dimensions cannot be greater than scan dimensions
        if len(kernel_size) > len(nav_shape):
            if kernel_size != (3, 3):
                warnings.warn(
                    f"Creates kernel of size {kernel_size[:len(nav_shape)]}, "
                    f"since input kernel size {kernel_size} has more dimensions"
                    f" than scan dimensions {nav_shape}."
                )
            kernel_size = kernel_size[: len(nav_shape)]

        # Kernel dimension cannot be greater than corresponding scan dimension
        if any(np.array(kernel_size) > np.array(nav_shape)):
            raise ValueError(
                f"Kernel size {kernel_size} is too large for a scan of "
                f"dimensions {nav_shape}."
            )

    # Get kernel from SciPy
    exclude_kernel_corners = False
    if isinstance(kernel, str):
        if kernel == "circular":
            exclude_kernel_corners = True
            kernel = "rectangular"

        # Pass any extra necessary parameters for kernel from SciPy
        window = (kernel,) + tuple(kwargs.values())
        returned_kernel = get_window(
            window=window, Nx=kernel_size[0], fftbins=False
        )

        # Add second dimension to kernel if kernel_size has two dimensions
        if len(kernel_size) == 2:
            returned_kernel = np.outer(
                returned_kernel,
                get_window(window=window, Nx=kernel_size[1], fftbins=False),
            )

    # If circular kernel, exclude kernel corners
    if exclude_kernel_corners and len(kernel_size) == 2:
        kernel_centre = np.array(kernel_size) // 2

        # Create an 'open' mesh-grid of the same size as the kernel
        y, x = np.ogrid[: kernel_size[0], : kernel_size[1]]
        distance_to_centre = np.sqrt(
            (x - kernel_centre[0]) ** 2 + (y - kernel_centre[0]) ** 2
        )
        mask = distance_to_centre > kernel_centre[0]
        returned_kernel[mask] = 0

    return returned_kernel


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

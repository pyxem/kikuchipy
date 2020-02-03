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

import matplotlib.pyplot as plt
import numbers
import numpy as np
from scipy.signal.windows import get_window
import warnings


def get_kernel(kernel="circular", kernel_size=(3, 3), axes=None, **kwargs):
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
    >>> kp.util.kernel.get_kernel(kernel="circular", kernel_size=(3, 3))
    array([[0., 1., 0.],
           [1., 1., 1.],
           [0., 1., 0.]])
    >>> kp.util.kernel.get_kernel(kernel="gaussian", std=2)
    array([[0.77880078, 0.8824969 , 0.77880078],
           [0.8824969 , 1.        , 0.8824969 ],
           [0.77880078, 0.8824969 , 0.77880078]])

    See Also
    --------
    plot_kernel
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


def plot_kernel(kernel, cmap="viridis", textcolors=None, ylabel="Coefficient"):
    """Plot a pattern kernel with indexed coefficients.

    Typically used to plot a pattern kernel returned by
    :func:`~kikuchipy.util.kernel.get_kernel`.

    Parameters
    ----------
    kernel : :class:`numpy.ndarray`
        Array, of max. two dimensions, with coefficients.
    cmap : :class:`matplotlib.colors.ListedColormap`, optional
    textcolors : list, optional
        A list or array of two color specifications.  The first is used
        for values below a threshold, the second for those above. If
        None (default), these are set to white and black.
    ylabel : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    image : matplotlib.image.AxesImage
    colorbar : matplotlib.colorbar.Colorbar

    See Also
    --------
    get_kernel

    """

    # Check if kernel is valid
    if kernel.ndim > 2:
        raise ValueError(
            "Can only plot a kernel of max. dimensions of 2, however a kernel "
            f"of {kernel.ndim} dimensions was passed."
        )

    fig, ax = plt.subplots()
    image = ax.imshow(kernel, cmap=cmap, interpolation=None)

    colorbar = ax.figure.colorbar(image, ax=ax)
    colorbar.ax.set_ylabel(ylabel, rotation=-90, va="bottom")

    # Set plot ticks
    ky, kx = kernel.shape
    ax.set_xticks(np.arange(kx))
    ax.set_yticks(np.arange(ky))
    ax.set_xticklabels(np.arange(kx) - int(np.floor(kx / 2)))
    ax.set_yticklabels(np.arange(ky) - int(np.floor(ky / 2)))

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(kx + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(ky + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    kw = dict(horizontalalignment="center", verticalalignment="center")
    threshold = image.norm(np.amax(kernel)) / 2.0
    if textcolors is None:
        textcolors = ["white", "black"]

    for i in range(kx):
        for j in range(ky):
            kw.update(
                color=textcolors[int(image.norm(kernel[j, i]) > threshold)]
            )
            coeff = kernel[j, i]
            coeff_str = str(round(coeff, 4) if coeff % 1 else int(coeff))
            image.axes.text(i, j, coeff_str, **kw)

    return fig, image, colorbar

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

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import get_window


class Kernel:
    """A kernel or mask of a given shape with coefficients.

    This class is a thin wrapper around a NumPy or Dask array with some
    additional convenience methods.

    Parameters
    ----------
    kernel : 'circular', 'rectangular', 'gaussian', str, \
            numpy.ndarray or dask.array.Array, optional
        Kernel type to create. Available types are listed in
        :func:`scipy.signal.windows.get_window`, in addition to a
        circular kernel (default) filled with ones in which corner
        coefficients are set to zero. A kernel element is considered to be
        in a corner if its radial distance to the origin (kernel centre) is
        shorter or equal to the half width of the kernel's longest axis. A
        1D or 2D :class:`numpy.ndarray` or :class:`dask.array.Array` can
        also be passed.
    kernel_size : sequence of ints, optional
        Size of the kernel. Not used if a custom kernel is passed to
        `kernel`. This can be either 1D or 2D, and can be asymmetrical.
        Default is (3, 3).
    **kwargs :
        Keyword arguments passed to the available kernel type listed in
        :func:`scipy.signal.windows.get_window`. If none are passed, the
        default values of that particular kernel are used.

    Attributes
    ----------
    type : str
        Name of kernel type. Can be either 'custom', 'circular',
        'rectangular', 'gaussian', or any other kernel type available via
        :func:`scipy.signal.windows.get_window`.
    coefficients : numpy.ndarray or dask.array.Array
        Kernel coefficients.
    circular : bool
        Whether the kernel is circular, with corner coefficients set to
        zero.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> import numpy as np

    The following passed parameters are the default

    >>> w = kp.util.Kernel(kernel="circular", kernel_size=(3, 3))
    >>> w
    circular, (3, 3)
    >>> w.coefficients
    array([[0., 1., 0.],
           [1., 1., 1.],
           [0., 1., 0.]])

    A kernel can be made circular

    >>> w = kp.util.Kernel(kernel="rectangular")
    >>> w
    rectangular, (3, 3)
    >>> w.coefficients
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])
    >>> w.make_circular()
    >>> w
    circular, (3, 3)
    >>> w.coefficients
    array([[0., 1., 0.],
           [1., 1., 1.],
           [0., 1., 0.]])

    A custom kernel can be created

    >>> w = kp.util.Kernel(np.arange(6).reshape(3, 2))
    >>> w
    custom, (3, 2)
    >>> w.coefficients
    array([[0, 1],
           [2, 3],
           [4, 5]])

    To create a Gaussian kernel with a standard deviation of 2, obtained
    from :func:`scipy.signal.windows.gaussian`

    >>> w = kp.util.Kernel(kernel="gaussian", std=2)
    >>> w
    gaussian, (3, 3)
    >>> w.coefficients
    array([[0.77880078, 0.8824969 , 0.77880078],
           [0.8824969 , 1.        , 0.8824969 ],
           [0.77880078, 0.8824969 , 0.77880078]])

    A plot of the kernel coefficients, with indices relative to the origin,
    can be produced and written to file

    >>> figure, image, colorbar = w.plot(cmap="inferno")
    >>> figure.savefig('my_kernel.png')

    See Also
    --------
    :func:`scipy.signal.windows.get_window`

    """

    def __init__(self, kernel="circular", kernel_size=(3, 3), **kwargs):
        # Turn into tuple
        kernel_size = tuple(kernel_size)

        # Class attributes
        self.type = None
        self.coefficients = None
        self.circular = False

        # Get kernel size if a custom kernel is passed, at the same time raising
        # an error if the custom kernel's shape is invalid
        if not isinstance(kernel, str):
            try:
                kernel_size = kernel.shape
                self.type = "custom"
                self.coefficients = kernel
            except AttributeError:
                raise ValueError(
                    f"Kernel {type(kernel)} must be of type numpy.ndarray or "
                    "dask.Array."
                )

        # Raise an error if kernel axes are fewer than one or if kernel_size is
        # invalid
        try:
            if any(np.array(kernel_size) < 1):
                raise ValueError(f"All kernel axes {kernel_size} must be > 0.")
            if any(isinstance(i, np.float) for i in np.array(kernel_size)):
                raise TypeError
        except TypeError:
            raise TypeError(
                f"Kernel size {kernel_size} must be a sequence of ints."
            )

        # Get kernel from SciPy if a custom one wasn't passed
        if isinstance(kernel, str):
            self.type = kernel
            if kernel == "circular":
                exclude_kernel_corners = True
                kernel = "rectangular"
            else:
                exclude_kernel_corners = False

            # Pass any extra necessary parameters for kernel from SciPy
            window = (kernel,) + tuple(kwargs.values())
            self.coefficients = get_window(
                window=window, Nx=kernel_size[0], fftbins=False
            )

            # Add second axis to kernel if kernel_size has two axes
            if len(kernel_size) == 2:
                self.coefficients = np.outer(
                    self.coefficients,
                    get_window(window=window, Nx=kernel_size[1], fftbins=False),
                )

            if exclude_kernel_corners:  # Exclude kernel corners
                self.make_circular()

    def __repr__(self):
        return f"{self.type}, {self.coefficients.shape}"

    def is_valid(self):
        """Return whether the kernel is in a valid state."""
        return (
            isinstance(self.type, str)
            and isinstance(self.coefficients, np.ndarray)
            and self.coefficients.ndim < 3
            and isinstance(self.circular, bool)
        )

    def make_circular(self):
        """Make kernel circular.

        The coefficients of kernel elements who's radial distance to the
        kernel origin is shorter or equal to the half width of the kernel's
        longest axis are set to zero. This has no effect if the kernel has
        only one axis.

        """
        if self.coefficients.ndim == 1:
            return

        # Get kernel size and origin
        ky, kx = self.coefficients.shape
        origin_y, origin_x = np.array(self.coefficients.shape) // 2

        # Create an 'open' mesh-grid of the same size as the kernel
        y, x = np.ogrid[:ky, :kx]
        distance_to_centre = np.sqrt((y - origin_y) ** 2 + (x - origin_x) ** 2)
        mask = distance_to_centre > max(origin_y, origin_x)

        # Update coefficients
        self.coefficients[mask] = 0
        self.circular = True

        # Update type
        if self.type in ["rectangular", "boxcar"]:
            self.type = "circular"

    def scan_compatible(self, scan):
        """Raise an error if the kernel is incompatible with the EBSD scan.

        Parameters
        ----------
        scan : EBSD or LazyEBSD
            An :class:`~kikuchipy.signals.ebsd.EBSD` or
            :class:`~kikuchipy.signals.ebsd.LazyEBSD` object.

        """
        try:
            nav_shape = scan.axes_manager.navigation_shape
        except AttributeError:
            raise AttributeError(
                f"Scan {type(scan)} must be a valid EBSD or LazyEBSD object."
            )

        # Number of kernel axes cannot be greater than scan axes, and a kernel
        # axis cannot be greater than the corresponding scan axis
        kernel_size = self.coefficients.shape
        if len(kernel_size) > len(nav_shape) or any(
            np.array(kernel_size) > np.array(nav_shape)
        ):
            raise ValueError(
                f"Kernel of size {kernel_size} is incompatible with the EBSD "
                f"scan of size {nav_shape}."
            )

    def _add_axes(self, n_dims):
        """Expand the shape of the coefficient array.

        Uses :func:`numpy.expand_dims`.

        Parameters
        ----------
        n_dims : int
            Number of axes to add.

        """
        for i in range(n_dims):
            self.coefficients = np.expand_dims(
                self.coefficients, axis=self.coefficients.ndim
            )

    def plot(self, cmap="viridis", textcolors=None, cmap_label="Coefficient"):
        """Plot kernel coefficients with indices relative to the origin.

        Parameters
        ----------
        cmap : str, optional
            A color map to color coefficients with, available in
            :class:`matplotlib.colors.ListedColormap`. Default is
            'viridis'.
        textcolors : list, optional
            A list of two color specifications. The first is used for
            values below a threshold, the second for those above. If None
            (default), these are set to white and black.
        cmap_label : str
            Color map label.

        Returns
        -------
        fig : matplotlib.figure.Figure
        image : matplotlib.image.AxesImage
        colorbar : matplotlib.colorbar.Colorbar

        """
        if not self.is_valid():
            raise ValueError("Kernel is invalid.")

        # Copy and use this object
        kernel = copy.copy(self)

        # Add axis if 1D
        if kernel.coefficients.ndim == 1:
            kernel._add_axes(1)

        fig, ax = plt.subplots()
        image = ax.imshow(kernel.coefficients, cmap=cmap, interpolation=None)

        colorbar = ax.figure.colorbar(image, ax=ax)
        colorbar.ax.set_ylabel(cmap_label, rotation=-90, va="bottom")

        # Set plot ticks
        ky, kx = kernel.coefficients.shape
        ax.set_xticks(np.arange(kx))
        ax.set_xticks(np.arange(kx + 1) - 0.5, minor=True)
        ax.set_xticklabels(np.arange(kx) - int(np.floor(kx / 2)))
        ax.set_yticks(np.arange(ky))
        ax.set_yticklabels(np.arange(ky) - int(np.floor(ky / 2)))
        ax.set_yticks(np.arange(ky + 1) - 0.5, minor=True)

        # Create grid
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Enter values of kernel coefficients as text
        kw = dict(horizontalalignment="center", verticalalignment="center")
        threshold = image.norm(np.amax(kernel.coefficients)) / 2.0
        if textcolors is None:
            textcolors = ["white", "black"]
        for idx in np.ndindex(kernel.coefficients.shape):
            coeff = kernel.coefficients[idx]
            kw.update(color=textcolors[int(image.norm(coeff) > threshold)])
            coeff_str = str(round(coeff, 4) if coeff % 1 else int(coeff))
            image.axes.text(idx[1], idx[0], coeff_str, **kw)

        return fig, image, colorbar

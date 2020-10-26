# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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

from copy import copy
from typing import List, Tuple, Optional, Sequence, Union

from dask.array import Array
from numba import njit
import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from matplotlib.pyplot import subplots
from scipy.signal.windows import get_window


class Window(np.ndarray):
    """A window/kernel/mask/filter of a given shape with some values.

    This class is a subclass of :class:`numpy.ndarray` with some
    additional convenience methods.

    It can be used to create a transfer function for filtering in the
    frequency domain, create an averaging window for averaging patterns
    with their nearest neighbours, and so on.

    Parameters
    ----------
    window : "circular", "rectangular", "gaussian", str, \
            numpy.ndarray, or dask.array.Array, optional
        Window type to create. Available types are listed in
        :func:`scipy.signal.windows.get_window` and includes
        "rectangular" and "gaussian", in addition to a
        "circular" window (default) filled with ones in which corner
        data are set to zero, a "modified_hann" window and "lowpass"
        and "highpass" FFT windows. A window element is considered to be
        in a corner if its radial distance to the origin (window centre)
        is shorter or equal to the half width of the windows's longest
        axis. A 1D or 2D :class:`numpy.ndarray` or
        :class:`dask.array.Array` can also be passed.
    shape : sequence of int, optional
        Shape of the window. Not used if a custom window is passed to
        `window`. This can be either 1D or 2D, and can be asymmetrical.
        Default is (3, 3).
    **kwargs :
        Keyword arguments passed to the window type. If none are
        passed, the default values of that particular window are used.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> import numpy as np

    The following passed parameters are the default

    >>> w = kp.filters.Window(window="circular", shape=(3, 3))
    >>> w
    Window (3, 3) circular
    [[0. 1. 0.]
     [1. 1. 1.]
     [0. 1. 0.]]

    A window can be made circular

    >>> w = kp.filters.Window(window="rectangular")
    >>> w
    Window (3, 3) rectangular
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    >>> w.make_circular()
    >>> w
    Window (3, 3) circular
    [[0. 1. 0.]
     [1. 1. 1.]
     [0. 1. 0.]]

    A custom window can be created

    >>> w = kp.filters.Window(np.arange(6).reshape(3, 2))
    >>> w
    Window (3, 2) custom
    [[0, 1]
     [2, 3]
     [4, 5]]

    To create a Gaussian window with a standard deviation of 2, obtained
    from :func:`scipy.signal.windows.gaussian`

    >>> w = kp.filters.Window(window="gaussian", std=2)
    >>> w
    Window (3, 3) gaussian
    [[0.77880078, 0.8824969 , 0.77880078]
     [0.8824969 , 1.        , 0.8824969 ]
     [0.77880078, 0.8824969 , 0.77880078]]

    See Also
    --------
    :func:`scipy.signal.windows.get_window`
    """

    name: str = None
    circular: bool = False

    def __new__(
        cls,
        window: Union[None, str, np.ndarray, Array] = None,
        shape: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        if window is None:
            window = "circular"

        if shape is None and "Nx" not in kwargs.keys():
            shape = (3, 3)
        elif "Nx" in kwargs.keys():
            shape = (kwargs.pop("Nx"),)
        else:  # Ensure valid shape tuple
            shape = tuple(shape)
            try:
                if any(np.array(shape) < 1):
                    raise ValueError(f"All window axes {shape} must be > 0.")
                if any(isinstance(i, np.float) for i in np.array(shape)):
                    raise TypeError
            except TypeError:
                raise TypeError(
                    f"Window shape {shape} must be a sequence of ints."
                )

        exclude_window_corners = False
        if isinstance(window, np.ndarray) or isinstance(window, Array):
            name = "custom"
            data = window
        elif isinstance(window, str):
            window_kwargs = {}

            if window == "modified_hann":
                name = window
                window_func = modified_hann
                window_kwargs["Nx"] = shape[0]
            elif window in ["lowpass", "highpass"]:
                name = window

                if window == "lowpass":
                    window_func = lowpass_fft_filter
                else:
                    window_func = highpass_fft_filter

                window_kwargs = {
                    "shape": shape,
                    "cutoff": kwargs["cutoff"],
                    "cutoff_width": kwargs.pop("cutoff_width", None),
                }
            else:  # Get window from SciPy
                if window == "circular":
                    exclude_window_corners = True
                    window = "rectangular"

                name = window
                window_func = get_window
                window_kwargs["fftbins"] = kwargs.pop("fftbins", False)
                window_kwargs["Nx"] = kwargs.pop("Nx", shape[0])
                window_kwargs["window"] = (window,) + tuple(kwargs.values())

            data = window_func(**window_kwargs)

            # Add second axis to window if shape has two axes
            if len(shape) == 2 and data.ndim == 1:
                window_kwargs["Nx"] = shape[1]
                data = np.outer(data, window_func(**window_kwargs))
        else:
            raise ValueError(
                f"Window {type(window)} must be of type numpy.ndarray, "
                "dask.array.Array or a valid string."
            )

        # Create object
        obj = np.asarray(data).view(cls)
        obj.name = name

        if exclude_window_corners:  # Exclude window corners
            obj.make_circular()

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, "name", None)
        self.circular = getattr(obj, "circular", False)

    def __array_wrap__(self, obj):
        if obj.shape == ():
            return obj[()]
        else:
            return np.ndarray.__array_wrap__(self, obj)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        shape = str(self.shape)
        name = self.name
        data = np.array_str(self, precision=4, suppress_small=True)
        return f"{cls} {shape} {name}\n{data}"

    @property
    def origin(self) -> tuple:
        """Window origin."""
        return tuple(i // 2 for i in self.shape)

    @property
    def distance_to_origin(self) -> np.ndarray:
        """Radial distance to the window origin."""
        return distance_to_origin(self.shape, self.origin)

    def make_circular(self):
        """Make window circular.

        The data of window elements who's radial distance to the
        window origin is shorter or equal to the half width of the
        window's longest axis are set to zero. This has no effect if the
        window has only one axis.
        """
        if self.ndim == 1:
            return

        # Get mask
        mask = self.distance_to_origin > max(self.origin)

        # Update data
        self[mask] = 0
        self.circular = True

        # Update name
        if self.name in ["rectangular", "boxcar"]:
            self.name = "circular"

    def is_valid(self) -> bool:
        """Return whether the window is in a valid state."""

        return (
            isinstance(self.name, str)
            and (isinstance(self, np.ndarray) or isinstance(self, Array))
            and self.ndim < 3
            and isinstance(self.circular, bool)
        )

    def shape_compatible(self, shape: Tuple[int]) -> bool:
        """Return whether window shape is compatible with a data shape.

        Parameters
        ----------
        shape
            Shape of data to apply window to.
        """
        # Number of window dimensions cannot be greater than data
        # dimensions, and a window axis cannot be greater than the
        # corresponding data axis
        window_shape = self.shape
        if len(window_shape) > len(shape) or any(
            np.array(window_shape) > np.array(shape)
        ):
            return False
        else:
            return True

    def plot(
        self,
        grid: bool = True,
        show_values: bool = True,
        cmap: str = "viridis",
        textcolors: Optional[List[str]] = None,
        cmap_label: str = "Value",
    ) -> Tuple[Figure, AxesImage, Colorbar]:
        """Plot window values with indices relative to the origin.

        Parameters
        ----------
        grid
            Whether to separate each value with a white spacing in a
            grid. Default is True.
        show_values
            Whether to show values as text in centre of element. Default
            is True.
        cmap
            A color map to color data with, available in
            :class:`matplotlib.colors.ListedColormap`. Default is
            "viridis".
        textcolors
            A list of two color specifications. The first is used for
            values below a threshold, the second for those above. If
            None (default), this is set to ["white", "black"].
        cmap_label
            Color map label. Default is "Value".

        Returns
        -------
        fig
        image
        colorbar

        Examples
        --------
        A plot of window data with indices relative to the origin,
        showing element values and x/y ticks, can be produced and
        written to file

        >>> figure, image, colorbar = w.plot(
        ...     cmap="inferno", grid=True, show_values=True)
        >>> figure.savefig('my_kernel.png')
        """
        if not self.is_valid():
            raise ValueError("Window is invalid.")

        # Copy and use this object
        w = copy(self)

        # Add axis if 1D
        if w.ndim == 1:
            w = np.expand_dims(w, axis=w.ndim)

        fig, ax = subplots()
        image = ax.imshow(w, cmap=cmap, interpolation=None)

        colorbar = ax.figure.colorbar(image, ax=ax)
        colorbar.ax.set_ylabel(cmap_label, rotation=-90, va="bottom")

        # Set plot ticks
        ky, kx = w.shape
        oy, ox = w.origin
        ax.set_xticks(np.arange(kx))
        ax.set_xticks(np.arange(kx + 1) - 0.5, minor=True)
        ax.set_xticklabels(np.arange(kx) - ox)
        ax.set_yticks(np.arange(ky))
        ax.set_yticklabels(np.arange(ky) - oy)
        ax.set_yticks(np.arange(ky + 1) - 0.5, minor=True)

        if grid:  # Create grid
            for edge, spine in ax.spines.items():
                spine.set_visible(False)
            ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
            ax.tick_params(which="minor", bottom=False, left=False)

        if show_values:
            # Enter values of window data as text
            kw = dict(horizontalalignment="center", verticalalignment="center")
            threshold = image.norm(np.amax(w)) / 2.0
            if textcolors is None:
                textcolors = ["white", "black"]
            for idx in np.ndindex(w.shape):
                val = w[idx]
                kw.update(color=textcolors[int(image.norm(val) > threshold)])
                coeff_str = str(round(val, 4) if val % 1 else int(val))
                image.axes.text(idx[1], idx[0], coeff_str, **kw)

        return fig, image, colorbar


def distance_to_origin(
    shape: Union[Tuple[int], Tuple[int, int]],
    origin: Union[None, Tuple[int], Tuple[int, int]] = None,
) -> np.ndarray:
    """Return the distance to the window origin in pixels.

    Parameters
    ----------
    shape
        Window shape.
    origin
        Window origin. If None, half the shape is used as origin for
        each axis.
    """
    if origin is None:
        origin = tuple(i // 2 for i in shape)

    coordinates = np.ogrid[tuple(slice(None, i) for i in shape)]
    if len(origin) == 2:
        squared = [(i - o) ** 2 for i, o in zip(coordinates, origin)]
        return np.sqrt(np.add.outer(*squared).squeeze())
    else:
        return abs(coordinates[0] - origin[0])


@njit
def modified_hann(Nx: int) -> np.ndarray:
    r"""Return a 1D modified Hann window with the maximum value
    normalized to 1.

    Used in [Wilkinson2006]_.

    Parameters
    ----------
    Nx
        Number of points in the window.

    Returns
    -------
    w : numpy.ndarray
        1D Hann window.

    Notes
    -----
    The modified Hann window is defined as

    .. math:: w(x) = \cos\left(\frac{\pi x}{N_x}\right),

    with :math:`x` relative to the window centre.

    References
    ----------
    .. [Wilkinson2006] A. J. Wilkinson, G. Meaden, D. J. Dingley, \
        "High resolution mapping of strains and rotations using \
        electron backscatter diffraction," Materials Science and \
        Technology 22(11), (2006), doi:
        https://doi.org/10.1179/174328406X130966.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> w1 = kp.filters.modified_hann(Nx=30)
    >>> w2 = kp.filters.Window("modified_hann", shape=(30,))
    >>> np.allclose(w1, w2)
    True
    """
    return np.cos(np.pi * (np.arange(Nx) - (Nx / 2) + 0.5) / Nx)


def lowpass_fft_filter(
    shape: Tuple[int, int],
    cutoff: Union[int, float],
    cutoff_width: Union[None, int, float] = None,
) -> np.ndarray:
    r"""Return a frequency domain low-pass filter transfer function in
    2D.

    Used in [Wilkinson2006]_.

    Parameters
    ----------
    shape
        Shape of function.
    cutoff
        Cut-off frequency.
    cutoff_width
        Width of cut-off region. If None (default), it is set to half of
        the cutoff frequency.

    Returns
    -------
    w : numpy.ndarray
        2D transfer function.

    Notes
    -----
    The low-pass filter transfer function is defined as

    .. math::

        w(r) = e^{-\left(\frac{r - c}{\sqrt{2}w_c/2}\right)^2},
        w(r) =
        \begin{cases}
        0, & r > c + 2w_c \\
        1, & r < c,
        \end{cases}

    where :math:`r` is the radial distance to the window centre,
    :math:`c` is the cut-off frequency, and :math:`w_c` is the width of
    the cut-off region.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> w1 = kp.filters.Window(
    ...     "lowpass", cutoff=30, cutoff_width=15, shape=(96, 96))
    >>> w2 = kp.filters.lowpass_fft_filter(
            shape=(96, 96), cutoff=30, cutoff_width=15)
    >>> np.allclose(w1, w2)
    True
    """
    r = distance_to_origin(shape)

    if cutoff_width is None:
        cutoff_width = cutoff / 2

    w = np.exp(-(((r - cutoff) / (np.sqrt(2) * cutoff_width / 2)) ** 2))
    w[r > (cutoff + (2 * cutoff_width))] = 0
    w[r < cutoff] = 1

    return w


def highpass_fft_filter(
    shape: Tuple[int, int],
    cutoff: Union[int, float],
    cutoff_width: Union[None, int, float] = None,
) -> np.ndarray:
    r"""Return a frequency domain high-pass filter transfer function in
    2D.

    Used in [Wilkinson2006]_.

    Parameters
    ----------
    shape
        Shape of function.
    cutoff
        Cut-off frequency.
    cutoff_width
        Width of cut-off region. If None (default), it is set to half of
        the cutoff frequency.

    Returns
    -------
    w : numpy.ndarray
        2D transfer function.

    Notes
    -----
    The high-pass filter transfer function is defined as

    .. math::

        w(r) = e^{-\left(\frac{c - r}{\sqrt{2}w_c/2}\right)^2},
        w(r) =
        \begin{cases}
        0, & r < c - 2w_c\\
        1, & r > c,
        \end{cases}

    where :math:`r` is the radial distance to the window centre,
    :math:`c` is the cut-off frequency, and :math:`w_c` is the width of
    the cut-off region.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> w1 = kp.filters.Window(
    ...     "highpass", cutoff=1, cutoff_width=0.5, shape=(96, 96))
    >>> w2 = kp.filters.highpass_fft_filter(
    ...     shape=(96, 96), cutoff=1, cutoff_width=0.5)
    >>> np.allclose(w1, w2)
    True
    """
    r = distance_to_origin(shape)

    if cutoff_width is None:
        cutoff_width = cutoff / 2

    w = np.exp(-(((cutoff - r) / (np.sqrt(2) * cutoff_width / 2)) ** 2))
    w[r < (cutoff - (2 * cutoff_width))] = 0
    w[r > cutoff] = 1

    return w

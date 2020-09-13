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

from copy import deepcopy
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


class EBSDDetector:
    def __init__(
        self,
        shape: Tuple[int, int] = (1, 1),
        px_size: float = 1,
        binning: int = 1,
        tilt: float = 0,
        sample_tilt: float = 70,
        pc: Union[np.ndarray, list, tuple] = (0.5, 0.5, 0.5),
        convention: Optional[str] = None,
    ):
        """Create an EBSD detector with a shape, pixel size, binning,
        and projection/pattern center(s) (PC(s)).

        Parameters
        ----------
        shape
            Number of detector rows and columns in pixels. Default is
            (1, 1).
        px_size
            Size of unbinned detector pixel in um, assuming a square
            pixel shape. Default is 1 um.
        binning
            Detector binning, i.e. how many pixels are binned into one.
            Default is 1, i.e. no binning.
        tilt
            Detector tilt from horizontal in degrees. Default is 0.
        sample_tilt
            Sample tilt from horizontal in degrees. Default is 70.
        pc
            X, Y and Z coordinates of the projection/pattern centers
            (PCs), describing the location of the beam on the sample
            measured relative to the detection screen. X and Y are
            measured from the detector left and top, respectively, while
            Z is the distance from the sample to the detection screen
            divided by the detector height. Default is (0.5, 0.5, 0.5).
        convention
            PC convention. If None (default), Bruker's convention is
            assumed. Options are "tsl", "oxford", "bruker", "emsoft",
            "emsoft4", and "emsoft5". "emsoft" and "emsoft5" is the same
            convention.
        """
        self.shape = shape
        self.px_size = px_size
        self.binning = binning
        self.tilt = tilt
        self.sample_tilt = sample_tilt
        self.pc = pc
        self._set_pc_convention(convention)

    @property
    def specimen_scintillator_distance(self) -> float:
        """Specimen to scintillator distance (SSD), also known as L."""
        return self.pcz * self.height

    @property
    def nrows(self) -> int:
        """Number of rows in pixels."""
        return self.shape[0]

    @property
    def ncols(self) -> int:
        """Number of columns in pixels."""
        return self.shape[1]

    @property
    def size(self) -> int:
        """Number of pixels."""
        return self.nrows * self.ncols

    @property
    def height(self) -> float:
        """Detector height in microns."""
        return self.nrows * self.px_size * self.binning

    @property
    def width(self) -> float:
        """Detector width in microns."""
        return self.ncols * self.px_size * self.binning

    @property
    def aspect_ratio(self) -> float:
        """Number of detector rows divided by columns."""
        return self.nrows / self.ncols

    @property
    def shape_unbinned(self) -> Tuple[int, int]:
        """Unbinned detector shape in pixels."""
        return tuple(np.array(self.shape) * self.binning)

    @property
    def px_size_binned(self) -> float:
        """Binned pixel size in microns."""
        return self.px_size * self.binning

    @property
    def pc(self) -> np.ndarray:
        """All projection center coordinates."""
        return self._pc

    @pc.setter
    def pc(self, value: Union[np.ndarray, List, Tuple]):
        """Set all projection center coordinates."""
        self._pc = np.atleast_2d(value)

    @property
    def pcx(self) -> np.ndarray:
        """Projection center x coordinates."""
        return self.pc[..., 0]

    @pcx.setter
    def pcx(self, value: Union[np.ndarray, list, tuple, float]):
        """Set the x projection center coordinates."""
        self._pc[..., 0] = np.atleast_2d(value)

    @property
    def pcy(self) -> np.ndarray:
        """Projection center y coordinates."""
        return self.pc[..., 1]

    @pcy.setter
    def pcy(self, value: Union[np.ndarray, list, tuple, float]):
        """Set y projection center coordinates."""
        self._pc[..., 1] = np.atleast_2d(value)

    @property
    def pcz(self) -> np.ndarray:
        """Projection center z coordinates."""
        return self.pc[..., 2]

    @pcz.setter
    def pcz(self, value: Union[np.ndarray, list, tuple, float]):
        """Set z projection center coordinates."""
        self._pc[..., 2] = np.atleast_2d(value)

    @property
    def pc_average(self) -> np.ndarray:
        """Return the overall average projection center."""
        ndim = self.pc.ndim
        axis = ()
        if ndim == 2:
            axis += (0,)
        elif ndim == 3:
            axis += (0, 1)
        return np.mean(self.pc, axis=axis).round(3)

    @property
    def navigation_shape(self) -> tuple:
        """Navigation shape of the projection center array."""
        return self.pc.shape[: self.pc.ndim - 1]

    @navigation_shape.setter
    def navigation_shape(self, value: tuple):
        """Set navigation shape of the projection center array."""
        ndim = len(value)
        if ndim > 2:
            raise ValueError(f"A maximum dimension of 2 is allowed, 2 < {ndim}")
        else:
            self.pc = self.pc.reshape(value + (3,))

    @property
    def navigation_dimension(self) -> int:
        """Number of navigation dimensions (a maximum of 2)."""
        return len(self.navigation_shape)

    @property
    def extent(self) -> np.ndarray:
        """Detector extent [x0, x1, y0, y1] in pixel coordinates."""
        return np.array([0, self.ncols, 0, self.nrows])

    @property
    def extent_gnomonic(self) -> np.ndarray:
        """Detector extent [x0, x1, y0, y1] in gnomonic coordinates."""
        return np.concatenate([self.x_range, self.y_range]).reshape(-1)

    @property
    def x_min(self) -> Union[np.ndarray, float]:
        """Left bound of detector in gnomonic projection."""
        return -self.aspect_ratio * (self.pcx / self.pcz)

    @property
    def x_max(self) -> Union[np.ndarray, float]:
        """Right bound of detector in gnomonic projection."""
        return self.aspect_ratio * (1 - self.pcx) / self.pcz

    @property
    def x_range(self) -> np.ndarray:
        """X detector limits in gnomonic projection."""
        # TODO: Decide whether we need dstack?
        return np.dstack((self.x_min, self.x_max))

    @property
    def y_min(self) -> Union[np.ndarray, float]:
        """Top bound of detector in gnomonic projection."""
        return -(1 - self.pcy) / self.pcz

    @property
    def y_max(self) -> Union[np.ndarray, float]:
        """Bottom bound of detector in gnomonic projection."""
        return self.pcy / self.pcz

    @property
    def y_range(self) -> np.ndarray:
        """Y detector limits in gnomonic projection."""
        return np.dstack((self.y_min, self.y_max))

    @property
    def x_scale(self) -> np.ndarray:
        """Width of a pixel in gnomonic projection."""
        if self.ncols == 1:
            return np.diff(self.x_range)
        else:
            return np.diff(self.x_range) / (self.ncols - 1)

    @property
    def y_scale(self) -> np.ndarray:
        """Height of a pixel in gnomonic projection."""
        if self.nrows == 1:
            return np.diff(self.y_range)
        else:
            return np.diff(self.y_range) / (self.nrows - 1)

    @property
    def r_max(self):
        """Maximum distance from PC to detector edge in gnomonic
        projection.
        """
        corners = np.zeros(self.navigation_shape + (4,))
        corners[..., 0] = self.x_min ** 2 + self.y_min ** 2  # Upper left
        corners[..., 1] = self.x_max ** 2 + self.y_min ** 2  # Upper right
        corners[..., 2] = self.x_max ** 2 + self.y_max ** 2  # Lower right
        corners[..., 3] = self.x_min ** 2 + self.y_min ** 2  # Lower left
        return np.sqrt(np.max(corners, axis=-1))

    def __repr__(self) -> str:
        """Nice string representation."""
        return (
            f"{self.__class__.__name__} {self.shape}, "
            f"px_size {self.px_size} um, binning {self.binning}, "
            f"tilt {self.tilt}, pc {tuple(self.pc_average)}"
        )

    def _set_pc_convention(self, convention: str):
        """Set appropriate PC based on vendor convention."""
        if convention is None or convention.lower() == "bruker":
            pass
        elif convention.lower() in ["tsl", "edax", "amatek"]:
            self.pc = self._pc_tsl2bruker()
        elif convention.lower() == "oxford":
            self.pc = self._pc_tsl2bruker()
        elif convention.lower() in ["emsoft", "emsoft4", "emsoft5"]:
            try:
                version = int(convention[-1])
            except ValueError:
                version = 5
            self.pc = self._pc_emsoft2bruker(version=version)
        else:
            conventions = [
                "bruker",
                "emsoft",
                "emsoft4",
                "emsoft5",
                "oxford",
                "tsl",
            ]
            raise ValueError(
                f"Projection center convention '{convention}' not among the "
                f"recognised conventions {conventions}."
            )

    def _pc_emsoft2bruker(self, version: int = 5) -> np.ndarray:
        new_pc = np.zeros_like(self.pc)
        if version == 5:
            new_pc[..., 0] = 0.5 + (-self.pcx / (self.ncols * self.binning))
        else:
            new_pc[..., 0] = 0.5 + (self.pcx / (self.ncols * self.binning))
        new_pc[..., 1] = 0.5 - (self.pcy / (self.nrows * self.binning))
        new_pc[..., 2] = self.pcz / (self.nrows * self.px_size * self.binning)
        return new_pc

    def _pc_tsl2bruker(self) -> np.ndarray:
        new_pc = deepcopy(self.pc)
        new_pc[..., 1] = 1 - self.pcy
        return new_pc

    def _pc_bruker2emsoft(self, version: int = 5) -> np.ndarray:
        new_pc = np.zeros_like(self.pc)
        new_pc[..., 0] = self.ncols * (self.pcx - 0.5)
        if version == 5:
            new_pc[..., 0] = -new_pc[..., 0]
        new_pc[..., 1] = self.nrows * (0.5 - self.pcy)
        new_pc[..., 2] = self.nrows * self.px_size * self.pcz
        return new_pc * self.binning

    def _pc_bruker2tsl(self) -> np.ndarray:
        new_pc = deepcopy(self.pc)
        new_pc[..., 1] = 1 - self.pcy
        return new_pc

    def pc_emsoft(self, version: int = 5) -> np.ndarray:
        """Return PC in the EMsoft convention.

        Parameters
        ----------
        version
            Which EMsoft PC convention to use. The direction of the x PC
            coordinate, xpc, flipped in version 5, because from then on
            the EBSD patterns were viewed looking from detector to
            sample, not the other way around.

        Returns
        -------
        np.ndarray
        """
        return self._pc_bruker2emsoft(version=version)

    def pc_bruker(self) -> np.ndarray:
        """Return PC in the Bruker convention."""
        return self.pc

    def pc_tsl(self) -> np.ndarray:
        """Return PC in the EDAX TSL convention."""
        return self._pc_bruker2tsl()

    def pc_oxford(self) -> np.ndarray:
        """Return PC in the Oxford convention."""
        return self._pc_bruker2tsl()

    def deepcopy(self):
        """Return a deep copy using :func:`copy.deepcopy`."""
        return deepcopy(self)

    def plot(
        self,
        coordinates: Optional[str] = None,
        show_pc: bool = True,
        pc_kwargs: Optional[dict] = None,
        pattern: Optional[np.ndarray] = None,
        pattern_kwargs: Optional[dict] = None,
        draw_gnomonic_circles: bool = False,
        gnomonic_angles: Union[None, list, np.ndarray] = None,
        gnomonic_circles_kwargs: Optional[dict] = None,
        zoom: float = 1,
        return_fig_ax: bool = False,
    ) -> Union[None, Tuple[plt.figure, plt.axis]]:
        """Plot the detector screen.

        Parameters
        ----------
        coordinates
            Which coordinates to use, "pixel" or "gnomonic". If None
            (default), "pixel" is used.
        show_pc
            Show the average projection center. Default is True.
        pc_kwargs
            A dictionary of keyword arguments passed to
            :meth:`matplotlib.axes.Axes.scatter`.
        pattern
            A pattern to put on the detector. If None (default), no
            pattern is displayed. The pattern array must have the
            same shape as the detector.
        pattern_kwargs
            A dictionary of keyword arguments passed to
            :meth:`matplotlib.axes.Axes.imshow`.
        draw_gnomonic_circles
            Draw circles for angular distances from pattern. Default is
            False. Circle positions are only correct when
            `coordinates="gnomonic"`.
        gnomonic_angles
            Which angular distances to plot if `draw_gnomonic_circles`
            is True. Default is from 10 to 80 in steps of 10.
        gnomonic_circles_kwargs
            A dictionary of keyword arguments passed to
            :meth:`matplotlib.patches.Circle`.
        zoom
            Whether to zoom in/out from the detector, e.g. to show the
            extent of the gnomonic projection circles. A zoom > 1 zooms
            out. Default is 1, i.e. no zoom.
        return_fig_ax
            Whether to return the figure and axis object created.
            Default is False.
        """
        sy, sx = self.shape
        pcx, pcy = self.pc_average[:2]

        if coordinates in [None, "pixel"]:
            pcy *= sy
            pcx *= sx
            extent = self.extent
            extent[2:] = extent[2:][::-1]
            x_label = r"$x_d$"
            y_label = r"$y_d$"
        else:
            pcy, pcx = (0, 0)
            extent = self.extent_gnomonic
            x_label = r"$x_g$"
            y_label = r"$y_g$"

        fig, ax = plt.subplots()
        ax.axis(zoom * extent)
        ax.set_aspect(self.aspect_ratio)
        ax.set_xlabel(x_label, fontsize=18)
        ax.set_ylabel(y_label, fontsize=18)

        # Plot a pattern on the detector
        if isinstance(pattern, np.ndarray):
            if pattern.shape != (sy, sx):
                raise ValueError(
                    f"Pattern shape {pattern.shape} must equal the detector "
                    f"shape {(sy, sx)}"
                )
            if pattern_kwargs is None:
                pattern_kwargs = {}
            pattern_kwargs.setdefault("cmap", "gray")
            ax.imshow(pattern, extent=extent, **pattern_kwargs)

        # Show the projection center
        if show_pc:
            if pc_kwargs is None:
                pc_kwargs = {}
            default_params_pc = {
                "s": 300,
                "facecolor": "gold",
                "edgecolor": "k",
                "marker": "*",
            }
            [pc_kwargs.setdefault(k, v) for k, v in default_params_pc.items()]
            ax.scatter(x=pcx, y=pcy, **pc_kwargs)

        # Draw gnomonic circles centered on the projection center
        if draw_gnomonic_circles:
            if gnomonic_circles_kwargs is None:
                gnomonic_circles_kwargs = {}
            default_params_gnomonic = {
                "alpha": 0.25,
                "edgecolor": "k",
                "facecolor": "None",
                "linewidth": 3,
            }
            [
                gnomonic_circles_kwargs.setdefault(k, v)
                for k, v in default_params_gnomonic.items()
            ]
            if gnomonic_angles is None:
                gnomonic_angles = np.arange(1, 9) * 10
            for angle in gnomonic_angles:
                ax.add_artist(
                    plt.Circle(
                        (pcx, pcy),
                        np.tan(np.deg2rad(angle)),
                        **gnomonic_circles_kwargs,
                    )
                )

        if return_fig_ax:
            return fig, ax

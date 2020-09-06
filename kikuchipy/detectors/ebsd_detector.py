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
            assumed.
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
        if self.ncols == 1:
            return np.diff(self.x_range)
        else:
            return np.diff(self.x_range) / (self.ncols - 1)

    @property
    def y_scale(self) -> np.ndarray:
        if self.nrows == 1:
            return np.diff(self.y_range)
        else:
            return np.diff(self.y_range) / (self.nrows - 1)

    @property
    def r_max(self):
        """Maximum distance from PC to detector edge."""
        corners = np.zeros(self.navigation_shape + (4,))
        corners[..., 0] = self.x_min ** 2 + self.y_min ** 2  # Upper left
        corners[..., 1] = self.x_max ** 2 + self.y_min ** 2  # Upper right
        corners[..., 2] = self.x_max ** 2 + self.y_max ** 2  # Lower right
        corners[..., 3] = self.x_min ** 2 + self.y_min ** 2  # Lower left
        return np.sqrt(np.max(corners, axis=-1))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} {self.shape}, "
            f"px_size {self.px_size} um, binning {self.binning}, "
            f"tilt {self.tilt}, pc {tuple(self.pc_average)}"
        )

    def _set_pc_convention(self, convention: str):
        """Set appropriate PC based on vendor convention."""
        if convention is None or convention == "bruker":
            pass
        elif convention.lower() == "tsl":
            self.pc = self._tsl2bruker()
        elif convention.lower() == "oxford":
            self.pc = self._oxford2emsoft()
            self.pc = self._emsoft2bruker()
        elif convention.lower() == "emsoft":
            self.pc = self._emsoft2bruker()
        else:
            conventions = ["bruker", "emsoft", "oxford", "tsl"]
            raise ValueError(
                f"Projection center convention '{convention}' not among the "
                f"recognised conventions {conventions}."
            )

    def _bruker2emsoft(self) -> np.ndarray:
        """Convert PC from Bruker to EMsoft convention."""
        new_pc = np.zeros_like(self.pc)
        new_pc[..., 0] = -self.ncols * (self.pcx - 0.5)
        new_pc[..., 1] = self.nrows * (0.5 - self.pcy)
        new_pc[..., 2] = self.nrows * self.px_size * self.pcz
        return new_pc

    def _emsoft2bruker(self) -> np.ndarray:
        """Convert PC from EMsoft to Bruker convention."""
        new_pc = np.zeros_like(self.pc)
        new_pc[..., 0] = (self.pcx / self.ncols) + 0.5
        new_pc[..., 1] = 0.5 - (self.pcy / self.nrows)
        new_pc[..., 2] = self.pcz / (self.ncols * self.px_size)
        return new_pc

    def _tsl2emsoft(self) -> np.ndarray:
        """Convert PC from EDAX TSL to EMsoft convention."""
        new_pc = np.zeros_like(self.pc)
        new_pc[..., 0] = -self.ncols * (self.pcx - 0.5)
        new_pc[..., 1] = self.nrows * (0.5 - self.pcy)
        new_pc[..., 2] = self.ncols * self.px_size * self.pcz
        return new_pc

    def _emsoft2tsl(self) -> np.ndarray:
        """Convert PC from EMsoft to EDAX TSL convention."""
        new_pc = np.zeros_like(self.pc)
        new_pc[..., 0] = (self.pcx / self.ncols) + 0.5
        new_pc[..., 1] = 0.5 - (self.pcy / self.nrows)
        new_pc[..., 2] = self.pcz / (self.ncols * self.px_size)
        return new_pc

    def _tsl2bruker(self) -> np.ndarray:
        """Convert PC from EDAX TSL to Bruker convention."""
        new_pc = self.pc[:]
        new_pc[..., 1] = 1 - new_pc[..., 1]
        return new_pc

    def _bruker2tsl(self) -> np.ndarray:
        """Convert PC from Bruker to EDAX TSL convention."""
        new_pc = self.pc[:]
        new_pc[..., 1] = 1 - new_pc[..., 1]
        return new_pc

    def _oxford2emsoft(self) -> np.ndarray:
        """Convert PC from Oxford to EMsoft convention."""
        new_pc = np.zeros_like(self.pc)
        new_pc[..., 0] = -self.ncols * (self.pcx - 0.5)
        new_pc[..., 1] = self.nrows * (self.pcy - 0.5)
        new_pc[..., 2] = self.ncols * self.px_size * self.pcz
        return new_pc

    def to_emsoft(self) -> np.ndarray:
        """Return PC in the EMsoft convention."""
        return self._bruker2emsoft()

    def to_bruker(self) -> np.ndarray:
        """Return PC in the Bruker convention."""
        return self.pc

    def to_tsl(self) -> np.ndarray:
        """Return PC in the EDAX TSL convention."""
        return self._bruker2tsl()

    def to_oxford(self) -> np.ndarray:
        """Return PC in the Oxford convention."""
        raise NotImplementedError

    def deepcopy(self):
        """Return a deep copy using :func:`copy.deepcopy`."""
        return deepcopy(self)

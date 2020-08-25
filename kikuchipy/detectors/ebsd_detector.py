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

from typing import Optional, Tuple

import numpy as np


class EBSDDetector:
    def __init__(
        self,
        rows: int,
        cols: int,
        pixel_size: float = 1,
        binning: int = 1,
        model: Optional[str] = None,
    ):
        """Create an EBSD detector with a shape, pixel size and pixel
        binning.


        Parameters
        ----------
        rows
            Number of rows, i.e. detector height, in pixels.
        cols
            Number of columns, i.e. detector width, in pixels.
        pixel_size
            Size of binned detector pixel in microns.
        binning
            Detector binning, i.e. how many pixels are binned into one.
        model
            Detector model.

        Examples
        --------
        >>> from kikuchipy import detectors
        >>> det = detectors.EBSDDetector(
        ...     rows=60,
        ...     cols=60,
        ...     pixel_size=70 * 8,
        ...     binning=8,
        ... )
        >>> det
        EBSDDetector (60, 60) px, px size 560.0 um, binning 8
        >>> det.shape_unbinned
        (480, 480)
        >>> det.aspect_ratio
        1.0
        >>> (det.height, det.width)
        (33600, 33600)
        """
        self.rows = rows
        self.cols = cols
        self.pixel_size = pixel_size
        self.binning = binning
        self.model = model

    @property
    def shape(self) -> Tuple[int, int]:
        """Detector shape in pixels."""
        return self.rows, self.cols

    @property
    def height(self) -> float:
        """Detector height in microns."""
        return self.rows * self.pixel_size

    @property
    def width(self) -> float:
        """Detector width in microns."""
        return self.cols * self.pixel_size

    @property
    def aspect_ratio(self) -> float:
        """Number of detector rows divided by columns."""
        return self.rows / self.cols

    @property
    def shape_unbinned(self) -> Tuple[int, int]:
        """Unbinned detector shape in pixels."""
        return tuple(np.array(self.shape) * self.binning)

    def __repr__(self):
        return (
            f"{self.__class__.__name__} {self.shape} px, px size "
            f"{self.pixel_size:.1f} um, binning {self.binning}"
        )

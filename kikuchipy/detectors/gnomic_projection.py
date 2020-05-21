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

from typing import Tuple


class Detector:
    def __init__(
        self, height: int, width: int, pixel_size: float = 1, binning: int = 1,
    ):
        self.height = height
        self.width = width
        self._pixel_size = pixel_size
        self.binning = binning

    def __repr__(self):
        return (
            "Detector: (h x w) = ({} x {}) px, px size: {:.1f} um, binning: {}"
        ).format(self.height, self.width, self.pixel_size, self.binning)

    @property
    def aspect_ratio(self):
        return self.width / self.height

    @property
    def pixel_size(self):
        return self._pixel_size * self.binning


class PatternCentre:
    def __init__(
        self,
        x: float = 1,
        y: float = 1,
        z: float = 1,
        convention: str = "emsoft",
        detector_shape: Tuple[int, int] = (1, 1),
        pixel_size: float = 1,
        binning: int = 1,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.detector = Detector(
            height=detector_shape[0],
            width=detector_shape[1],
            pixel_size=pixel_size,
            binning=binning,
        )
        self.convention = convention
        if convention == "tsl":
            self.x, self.y, self.z = self._tsl2emsoft()
        elif convention == "bruker":
            self.x, self.y, self.z = self._bruker2emsoft()
        elif convention == "oxford":
            self.x, self.y, self.z = self._oxford2emsoft()

    def __repr__(self):
        return "(PCx, PCy, PCz) = ({:.4f}, {:.4f}, {:.4f})".format(
            self.x, self.y, self.z
        )

    def _bruker2emsoft(self):
        """Convert pattern centre from Bruker to EMsoft convention."""
        new_x = self.detector.width * (self.x - 0.5)
        new_y = self.detector.height * (0.5 - self.y)
        new_z = self.detector.height * self.detector.pixel_size * self.z
        return new_x, new_y, new_z

    def _emsoft2bruker(self):
        """Convert pattern centre from EMsoft to Bruker convention."""
        new_x = (self.x / self.detector.width) + 0.5
        new_y = 0.5 - (self.y / self.detector.height)
        new_z = self.z / (self.detector.height * self.detector.pixel_size)
        return new_x, new_y, new_z

    def _tsl2emsoft(self):
        """Convert pattern centre from EDAX TSL to EMsoft convention."""
        new_x = self.detector.width * (self.x - 0.5)
        new_y = self.detector.height * (0.5 - self.y)
        new_z = self.detector.width * self.detector.pixel_size * self.z
        return new_x, new_y, new_z

    def _emsoft2tsl(self):
        """Convert pattern centre from EMsoft to EDAX TSL convention."""
        new_x = (self.x / self.detector.width) + 0.5
        new_y = 0.5 - (self.y / self.detector.height)
        new_z = self.z / (self.detector.width * self.detector.pixel_size)
        return new_x, new_y, new_z

    def _tsl2bruker(self):
        """Convert pattern centre from EDAX TSL to Bruker convention."""
        return 1 - self.x, self.y, self.z

    def _bruker2tsl(self):
        """Convert pattern centre from Bruker to EDAX TSL convention."""
        return 1 - self.x, self.y, self.z

    def _oxford2emsoft(self):
        """Convert pattern centre from Oxford to EMsoft convention."""
        new_x = self.detector.width * (self.x - 0.5)
        new_y = self.detector.height * (self.y - 0.5)
        new_z = self.detector.width * self.detector.pixel_size * self.z
        return new_x, new_y, new_z

    def to_emsoft(self):
        """Return the pattern centre in the EMsoft convention."""
        return self.x, self.y, self.z

    def to_bruker(self):
        """Return the pattern centre in the Bruker convention."""
        return self._emsoft2bruker()

    def to_tsl(self):
        """Convert pattern centre to EDAX TSL convention."""
        return self._emsoft2tsl()

    def to_oxford(self):
        """Convert pattern centre to Oxford convention."""
        raise NotImplementedError


class GnomicProjection:
    def __init__(self, pc: PatternCentre):
        self.pc = pc

    @property
    def y_range(self):
        """Screen limits in y direction."""
        pc = self.pc
        detector = pc.detector
        y_min = -(detector.pixel_size / pc.z) * (pc.y + 0.5 * detector.height)
        y_max = (detector.pixel_size / pc.z) * (0.5 * detector.height - pc.y)
        return y_min, y_max

    @property
    def x_range(self):
        """Screen limits in x direction."""
        pc = self.pc
        detector = pc.detector
        pre_factor = (
            detector.height
            * (detector.pixel_size / pc.z)
            * detector.aspect_ratio
        )
        x_min = -pre_factor * (0.5 + (pc.x / detector.width))
        x_max = pre_factor * (0.5 - (pc.x / detector.width))
        return x_min, x_max

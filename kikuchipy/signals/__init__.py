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

"""Experimental and simulated diffraction patterns and virtual
backscatter electron images.
"""

from kikuchipy.signals import util
from kikuchipy.signals.ebsd import EBSD, LazyEBSD
from kikuchipy.signals.ebsd_master_pattern import (
    EBSDMasterPattern,
    LazyEBSDMasterPattern,
)
from kikuchipy.signals.virtual_bse_image import VirtualBSEImage

__all__ = [
    "EBSD",
    "EBSDMasterPattern",
    "LazyEBSD",
    "LazyEBSDMasterPattern",
    "util",
    "VirtualBSEImage",
]

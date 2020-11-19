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

# Import order must not be changed
from kikuchipy import crystallography
from kikuchipy import detectors
from kikuchipy import draw
from kikuchipy import filters
from kikuchipy import indexing
from kikuchipy import pattern
from kikuchipy import projections
from kikuchipy import signals
from kikuchipy import generators
from kikuchipy import simulations
from kikuchipy.io._io import load
from kikuchipy import data  # Must be below io.load

from kikuchipy import release

__version__ = release.version

__all__ = [
    "crystallography",
    "data",
    "detectors",
    "draw",
    "filters",
    "generators",
    "indexing",
    "load",
    "pattern",
    "projections",
    "signals",
    "simulations",
]

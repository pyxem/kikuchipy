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

"""
Processing and analysis of electron backscatter diffraction (EBSD)
patterns in Python.

The ``kikuchipy`` library builds on the tools for multi-dimensional data
analysis provided by the HyperSpy library. This means that the EBSD and
EBSDMasterPattern classes, which have several common methods for
processing of EBSD patterns, also inherits all relevant methods from
HyperSpy's Signal2D and Signal classes.

Sub-packages
-----------
filters
    Transfer functions, spatial kernels, etc.
pattern
    Processing of a single pattern or chunks of patterns. Used by the
    signal classes.
signals
    Classes to store, process, analyze and save EBSD patterns and master
    patterns.

Utility functions
-----------------
load
    Reading of EBSD patterns and master patterns from file into signal
    objects.

"""

# Import order must not be changed
from kikuchipy import filters
from kikuchipy import pattern
from kikuchipy import signals
from kikuchipy import generators
from kikuchipy.io._io import load

from kikuchipy import release

__version__ = release.version

__all__ = [
    "filters",
    "generators",
    "load",
    "pattern",
    "signals",
]

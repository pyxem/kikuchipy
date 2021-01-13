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

"""Tools for indexing of EBSD patterns by comparison to simulated
patterns.

The EBSD method :meth:`~kikuchipy.signals.EBSD.match_patterns` uses
these tools for pattern matching.
"""

from kikuchipy.indexing._merge_crystal_maps import merge_crystal_maps
from kikuchipy.indexing.orientation_similarity_map import (
    orientation_similarity_map,
)
from kikuchipy.indexing import similarity_metrics
from kikuchipy.indexing._static_pattern_matching import StaticPatternMatching

__all__ = [
    "merge_crystal_maps",
    "orientation_similarity_map",
    "similarity_metrics",
    "StaticPatternMatching",
]

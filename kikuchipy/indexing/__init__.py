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

"""Tools for indexing of EBSD patterns by matching to a dictionary of
simulated patterns.

The EBSD method :meth:`~kikuchipy.signals.EBSD.dictionary_indexing` uses
some of these tools for dictionary indexing.
"""

from kikuchipy.indexing._merge_crystal_maps import merge_crystal_maps
from kikuchipy.indexing.orientation_similarity_map import orientation_similarity_map
from kikuchipy.indexing._refinement._refinement import (
    compute_refine_orientation_results,
    compute_refine_orientation_projection_center_results,
    compute_refine_projection_center_results,
)
from kikuchipy.indexing import similarity_metrics

__all__ = [
    "compute_refine_orientation_results",
    "compute_refine_orientation_projection_center_results",
    "compute_refine_projection_center_results",
    "merge_crystal_maps",
    "orientation_similarity_map",
    "similarity_metrics",
]

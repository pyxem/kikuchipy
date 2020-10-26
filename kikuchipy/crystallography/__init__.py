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

"""Crystallographic computations not found (easily) elsewhere."""

from kikuchipy.crystallography.matrices import (
    get_direct_structure_matrix,
    get_reciprocal_structure_matrix,
    get_reciprocal_metric_tensor,
)

__all__ = [
    "get_direct_structure_matrix",
    "get_reciprocal_structure_matrix",
    "get_reciprocal_metric_tensor",
]

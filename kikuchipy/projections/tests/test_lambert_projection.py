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

import numpy as np
import pytest

from kikuchipy.projections.lambert_projection import LambertProjection
from orix.vector import Vector3d


class TestLambertProjection:
    def test_project_Vector3d(self):
        vector_one = Vector3d((1, 2, 3))
        clsObjA = LambertProjection.project(vector_one)

        # !! WORK IN PROGRESS !!

        assert clsObjA[..., 0][0] == -1
        assert clsObjA[..., 1][0] == -1

        vector_two = Vector3d(np.array([[2, 1, 1], [3, 1, 1]]))
        clsObjB = LambertProjection.project(vector_two)
        pass

    def test_project_ndarray(self):
        pass

    def test_iproject(self):
        pass

    def test_eq_c(self):
        pass

    def test_lambert_to_gnomonic(self):
        pass

    def test_gnomonic_to_lambert(self):
        pass

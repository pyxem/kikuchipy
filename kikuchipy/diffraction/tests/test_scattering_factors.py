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

from kikuchipy.diffraction import atomic_scattering_parameters


class TestScatteringFactors:
    @pytest.mark.parametrize("element", ["Ni", "ni", 28])
    def test_atomic_scattering_parameters(self, element):
        desired_a = np.array([2.210, 2.134, 1.689, 0.524])
        desired_b = np.array([58.727, 13.553, 2.609, 0.339])
        a, b = atomic_scattering_parameters(element=element)
        assert np.allclose(a, desired_a)
        assert np.allclose(b, desired_b)

    @pytest.mark.parametrize(
        "unit, factor",
        [("Å", 1), ("a", 1), ("nm", 1e-2), ("NM", 1e-2)],
    )
    def test_atomic_scattering_parameters(self, unit, factor):
        a, b = atomic_scattering_parameters(element="Tc", unit=unit)
        desired_a = np.array([4.318, 3.270, 1.287, 0]) * factor
        desired_b = np.array([28.246, 5.148, 0.59, 0]) * factor
        assert np.allclose(a, desired_a)
        assert np.allclose(b, desired_b)

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

import numpy as np
from orix import quaternion, vector

import kikuchipy as kp


class TestRotationVectorTools:
    def test_rotate_vector(self):
        """Make sure the Numba function is covered and returns the same
        result as orix.
        """
        rot = np.array([0.7071, 0.7071, 0, 0])
        dc = kp.signals.util._master_pattern._get_direction_cosines_for_single_pc.py_func(
            pcx=0.5,
            pcy=0.5,
            pcz=0.5,
            nrows=480,
            ncols=640,
            tilt=10,
            azimuthal=0,
            sample_tilt=70,
        )
        dc = dc.reshape((-1, 3))
        rotated_dc = kp._rotation._rotate_vector.py_func(rot, dc)

        rot_orix = quaternion.Rotation(rot)
        dc_orix = vector.Vector3d(dc)
        rotated_dc_orix = rot_orix * dc_orix

        assert np.allclose(rotated_dc, rotated_dc_orix.data, atol=1e-3)

    def test_rotation_from_euler(self):
        euler = np.array([1, 2, 3])
        rot_orix = quaternion.Rotation.from_euler(euler).data
        assert np.allclose(rot_orix, kp._rotation._rotation_from_euler.py_func(*euler))

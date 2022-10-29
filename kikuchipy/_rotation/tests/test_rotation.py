# Copyright 2019-2022 The kikuchipy developers
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
from orix.quaternion import Rotation
from orix.vector import Rodrigues, Vector3d

import kikuchipy as kp


class TestRotationVectorTools:
    def test_rotate_vector(self):
        """Make sure the Numba function is covered and returns the same
        result as orix.
        """
        rot = np.array([0.7071, 0.7071, 0, 0])
        sig_shape = (20, 30)
        dc = (
            kp.signals.util._master_pattern._get_direction_cosines_for_fixed_pc.py_func(
                pcx=0.5,
                pcy=0.5,
                pcz=0.5,
                nrows=sig_shape[0],
                ncols=sig_shape[1],
                tilt=10,
                azimuthal=0,
                sample_tilt=70,
                mask=np.ones(sig_shape[0] * sig_shape[1], dtype=bool),
            )
        )

        rot_orix = Rotation(rot)
        dc_orix = Vector3d(dc)
        rotated_dc_orix = rot_orix * dc_orix

        rotated_dc = kp._rotation._rotate_vector(rot, dc)
        rotated_dc_py = kp._rotation._rotate_vector.py_func(rot, dc)

        assert np.allclose(rotated_dc, rotated_dc_py, atol=1e-3)
        assert np.allclose(rotated_dc_py, rotated_dc_orix.data, atol=1e-3)

    def test_rotation_from_rodrigues(self):
        rod = np.array([1, 2, 3])
        rot_orix = Rotation.from_neo_euler(Rodrigues(rod)).data
        rot_numba = kp._rotation._rotation_from_rodrigues(*rod)
        rot_numba_py = kp._rotation._rotation_from_rodrigues.py_func(*rod)

        assert np.allclose(rot_numba, rot_numba_py)
        assert np.allclose(rot_numba_py, rot_orix)

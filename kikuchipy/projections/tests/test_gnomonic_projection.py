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

from orix.vector import Vector3d
import numpy as np

from kikuchipy.projections.gnomonic_projection import GnomonicProjection


class TestGnomonicProjection:
    def test_project(self):
        """Projecting cartesian coordinates to Gnomonic coordinates
        yields expected result.
        """
        v = np.array(
            [
                [0.5901, 0.8439, 0.4139],
                [0.4994, 0.744, 0.6386],
                [0.8895, 0.7788, 0.5597],
                [0.6673, 0.8619, 0.6433],
                [0.7605, 0.0647, 0.9849],
                [0.5852, 0.3946, 0.5447],
                [0.4647, 0.7181, 0.6571],
                [0.8806, 0.5106, 0.6244],
                [0.0816, 0.4489, 0.0296],
                [0.7585, 0.1885, 0.2678],
            ]
        )
        xy = GnomonicProjection.project(v)

        # Desired project result?
        assert np.allclose(
            xy,
            np.array(
                [
                    [1.42, 2.03],
                    [0.78, 1.16],
                    [1.58, 1.39],
                    [1.03, 1.33],
                    [0.77, 0.06],
                    [1.07, 0.72],
                    [0.70, 1.09],
                    [1.41, 0.81],
                    [2.75, 15.14],
                    [2.83, 0.70],
                ]
            ),
            atol=1e-1,
        )

        # Same result passing Vector3d
        assert np.allclose(xy, GnomonicProjection.project(Vector3d(v)))

    def test_iproject(self):
        """Projecting Gnomonic coordinates to cartesian coordinates
        yields expected result.
        """
        xy = np.array(
            [
                [1.4254, 2.0385],
                [0.7819, 1.1650],
                [1.5892, 1.3915],
                [1.0372, 1.3397],
                [0.7721, 0.0656],
                [1.0743, 0.7245],
                [0.7072, 1.0928],
                [1.4103, 0.8177],
                [2.7529, 15.1496],
                [2.8328, 0.7039],
            ]
        )

        assert np.allclose(
            GnomonicProjection.iproject(xy).data,
            np.array(
                [
                    [0.5316, 0.7603, 0.3729],
                    [0.4538, 0.6761, 0.5803],
                    [0.6800, 0.5954, 0.4278],
                    [0.5272, 0.6809, 0.5082],
                    [0.6103, 0.0519, 0.7904],
                    [0.6563, 0.4426, 0.6109],
                    [0.4308, 0.6657, 0.6091],
                    [0.7374, 0.4275, 0.5228],
                    [0.1784, 0.9818, 0.06480],
                    [0.9181, 0.2281, 0.3240],
                ]
            ),
            atol=1e-2,
        )

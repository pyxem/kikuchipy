# Copyright 2019-2024 The kikuchipy developers
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

import kikuchipy as kp
from kikuchipy._utils._gnonomic_bounds import get_gnomonic_bounds


class TestGnomonicBounds:
    s = kp.data.nickel_ebsd_small()
    det_1d = kp.detectors.EBSDDetector(shape=(1024, 1244), pc=s.detector.pc[0, 0])
    nrows = det_1d.nrows
    ncols = det_1d.ncols
    pcx = det_1d.pcx[0]
    pcy = det_1d.pcy[0]
    pcz = det_1d.pcz[0]

    def test_gnomonic_bounds(self):
        gn_b = get_gnomonic_bounds(self.nrows, self.ncols, self.pcx, self.pcy, self.pcz)

        assert np.allclose(gn_b, np.squeeze(self.det_1d.gnomonic_bounds))

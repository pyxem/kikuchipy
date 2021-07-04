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
import pytest

from kikuchipy.indexing._refinement import (
    _py_ncc,
    _fast_get_lambert_interpolation_parameters,
    _fast_lambert_projection,
    _fast_simulate_single_pattern,
    _fast_get_dc,
    _fast_get_dc_multiple_pc,
)


class TestEBSDRefinementMethods:

    # Added .py_func to the JIT functions
    def test_ncc(self):
        a = np.random.rand(100, 100)
        assert _py_ncc.py_func(a, a) == pytest.approx(1.0)

    def test_get_dc_multiple_pc(self):
        xpc = np.array((0.5, 0.4))
        ypc = np.array((0.5, 0.4))
        L = np.array((200, 300))
        scan_points = 2
        ncols = 60
        nrows = 60
        px_size = 1
        alpha = 0.34

        out = _fast_get_dc_multiple_pc.py_func(
            xpc, ypc, L, scan_points, ncols, nrows, px_size, alpha
        )

        assert out.shape == (scan_points, nrows, ncols, 3)

    def test_get_dc(self):
        xpc = 0.5
        ypc = 0.5
        L = 200
        ncols = 60
        nrows = 60
        px_size = 1
        alpha = 0.34

        out = _fast_get_dc.py_func(xpc, ypc, L, ncols, nrows, px_size, alpha)

        assert out.shape == (nrows, ncols, 3)

    def test_single_pattern_sim(self):
        r = np.array((1, 0, 0, 0))
        dc = np.random.rand(60, 60, 3)
        master_north = np.random.rand(401, 401)
        master_south = np.random.rand(401, 401)
        npx = 401
        npy = 401
        scale = 1

        out1 = _fast_simulate_single_pattern.py_func(
            r, dc, master_north, master_south, npx, npy, scale
        )

        assert out1.shape == (60, 60)

        out2 = _fast_simulate_single_pattern.py_func(
            r, -dc, master_north, master_south, npx, npy, scale
        )

        assert out2.shape == (60, 60)

    def test_lambert_params(self):
        rdc = np.random.rand(60, 60)
        npx = 401
        npy = 401
        scale = 1
        a, b, c, d, e, f, g, h = _fast_get_lambert_interpolation_parameters.py_func(
            rdc, npx, npy, scale
        )

        single_shape = (60,)
        assert a.shape == single_shape
        assert b.shape == single_shape
        assert c.shape == single_shape
        assert d.shape == single_shape
        assert e.shape == single_shape
        assert f.shape == single_shape
        assert g.shape == single_shape
        assert h.shape == single_shape

    def test_lambert_projection(self):
        v = np.random.rand(60, 60, 3)
        w = _fast_lambert_projection.py_func(v)
        assert w.shape == (60, 60, 2)

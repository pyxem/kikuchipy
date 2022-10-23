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
from orix.crystal_map import Phase
import pytest

import kikuchipy as kp


class TestEBSDRefinementSetup:
    @pytest.mark.parametrize(
        "method, method_kwargs", [("minimize", None), ("differential_evolution", None)]
    )
    def test_get_optimization_method_with_kwargs(self, method, method_kwargs):
        (
            method_out,
            method_kwargs_out,
        ) = kp.indexing._refinement._refinement._get_optimization_method_with_kwargs(
            method, method_kwargs
        )
        assert callable(method_out)
        assert method_out.__name__ == method
        assert isinstance(method_kwargs_out, dict)

        if method == "minimize":
            assert method_kwargs_out["method"] == "Nelder-Mead"

    def test_get_optimization_method_with_kwargs_raises(self):
        method = "wait-and-see"
        with pytest.raises(ValueError, match=f"Method {method} not in the list of"):
            _ = kp.indexing._refinement._refinement._get_optimization_method_with_kwargs(
                method
            )

    def test_check_master_pattern_and_get_data(self):
        axes = [
            dict(name="hemisphere", size=2, scale=1),
            dict(name="energy", size=5, offset=16, scale=1),
            dict(name="dy", size=5, scale=1),
            dict(name="dx", size=5, scale=1),
        ]
        mp_data = np.random.rand(2, 5, 5, 5).astype(np.float64)
        mp = kp.signals.EBSDMasterPattern(
            mp_data,
            axes=axes,
            projection="lambert",
            hemisphere="both",
            phase=Phase("ni", 225),
        )
        (
            mpn,
            mps,
            npx,
            npy,
            scale,
        ) = kp.indexing._refinement._refinement._check_master_pattern_and_get_data(
            master_pattern=mp,
            energy=20,
        )

        assert npx == mp_data.shape[3]
        assert npy == mp_data.shape[2]
        assert scale == (mp_data.shape[3] - 1) / 2

        # Master patterns are rescaled
        assert mpn.dtype == np.float32
        assert not np.allclose(mpn, mp_data[0, -1])
        assert not np.allclose(mps, mp_data[1, -1])
        assert np.min(mpn) >= -1
        assert np.max(mpn) <= 1


class TestRefinementSolverNumba:
    def test_mask_pattern_numba(self):
        p = np.arange(100).astype("float32")
        mask = np.ones(p.size, dtype=bool)
        mask[:10] = False
        p_masked = kp.indexing._refinement._solvers._mask_pattern(p, mask)
        p_masked2 = kp.indexing._refinement._solvers._mask_pattern.py_func(p, mask)
        assert p_masked.size == 90
        assert np.isclose(p_masked.mean(), 54.5)
        assert np.allclose(p_masked, p_masked2)

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

import os

import dask.array as da
import numpy as np
from orix.crystal_map import Phase
import pytest

import kikuchipy as kp


DIR_PATH = os.path.dirname(__file__)
ECP_FILE = os.path.join(
    DIR_PATH, "../../data/emsoft_ecp_master_pattern/ecp_master_pattern.h5"
)


class TestECPMasterPattern:
    def test_init_no_metadata(self):
        s = kp.signals.ECPMasterPattern(
            np.zeros((2, 10, 11, 11)),
            projection="lambert",
            hemisphere="both",
            phase=Phase("a"),
        )
        assert isinstance(s.phase, Phase)
        assert s.phase.name == "a"
        assert s.projection == "lambert"
        assert s.hemisphere == "both"

    def test_init_lazy_ecp_master_pattern(self):
        s = kp.signals.LazyECPMasterPattern(da.zeros((2, 10, 11, 11)))
        assert isinstance(s, kp.signals.LazyECPMasterPattern)
        assert isinstance(s.data, da.Array)
        s.compute()
        assert isinstance(s, kp.signals.ECPMasterPattern)

    def test_set_custom_properties(self):
        s = kp.load(ECP_FILE)

        # phase
        s.phase = Phase("b")
        assert s.phase.name == "b"

        # projection
        s.projection = "spherical"
        assert s.projection == "spherical"

        # hemisphere
        s.hemisphere = "east"
        assert s.hemisphere == "east"

    def test_get_master_pattern_arrays_from_energy(self):
        """Get upper and lower hemisphere of master pattern of the
        last energy axis without providing the energy parameter.
        """
        shape = (2, 11, 11)
        data = np.arange(np.prod(shape)).reshape(shape)
        mp = kp.signals.EBSDMasterPattern(
            data,
            axes=[
                {"size": 2, "name": "energy"},
                {"size": 11, "name": "x"},
                {"size": 11, "name": "y"},
            ],
        )
        mp_upper, mp_lower = mp._get_master_pattern_arrays_from_energy()
        assert np.allclose(mp_upper, data[1])
        assert np.allclose(mp_lower, data[1])

    @pytest.mark.skipif(not kp._pyvista_installed, reason="PyVista is not installed")
    def test_plot_spherical(self):
        """Cover inherited method only included for documentation
        purposes (tested rigorously elsewhere).
        """
        s = kp.load(ECP_FILE)
        s.plot_spherical()

    def test_inherited_methods(self):
        """Cover inherited method only included for documentation
        purposes (tested rigorously elsewhere).
        """
        s = kp.load(ECP_FILE)

        # as_lambert()
        s2 = s.as_lambert()
        assert s2.projection == "lambert"

        # normalize_intensity()
        s.normalize_intensity()

        # rescale_intensity()
        s.rescale_intensity()

        # deepcopy()
        s3 = s.deepcopy()
        assert not np.may_share_memory(s.data, s3.data)

# Copyright 2019-2023 The kikuchipy developers
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

    def test_rescale_intensity_inplace(self):
        mp = kp.load(ECP_FILE)

        # Current signal is unaffected
        mp2 = mp.deepcopy()
        mp3 = mp.normalize_intensity(inplace=False)
        assert isinstance(mp3, kp.signals.ECPMasterPattern)
        assert np.allclose(mp2.data, mp.data)

        # Operating on current signal gives same result as output
        mp.normalize_intensity()
        assert np.allclose(mp3.data, mp.data)

        # Operating on lazy signal returns lazy signal
        mp4 = mp2.as_lazy()
        mp5 = mp4.normalize_intensity(inplace=False)
        assert isinstance(mp5, kp.signals.LazyECPMasterPattern)
        mp5.compute()
        assert np.allclose(mp5.data, mp.data)

    def test_rescale_intensity_lazy_output(self):
        mp = kp.load(ECP_FILE)
        with pytest.raises(
            ValueError, match="`lazy_output=True` requires `inplace=False`"
        ):
            _ = mp.normalize_intensity(lazy_output=True)

        mp2 = mp.normalize_intensity(inplace=False, lazy_output=True)
        assert isinstance(mp2, kp.signals.LazyECPMasterPattern)

        mp3 = mp.as_lazy()
        mp4 = mp3.normalize_intensity(inplace=False, lazy_output=False)
        assert isinstance(mp4, kp.signals.ECPMasterPattern)

    def test_normalize_intensity_inplace(self):
        mp = kp.load(ECP_FILE)

        # Current signal is unaffected
        mp2 = mp.deepcopy()
        mp3 = mp.normalize_intensity(inplace=False)
        assert isinstance(mp3, kp.signals.ECPMasterPattern)
        assert np.allclose(mp2.data, mp.data)

        # Operating on current signal gives same result as output
        mp.normalize_intensity()
        assert np.allclose(mp3.data, mp.data)

        # Operating on lazy signal returns lazy signal
        mp4 = mp2.as_lazy()
        mp5 = mp4.normalize_intensity(inplace=False)
        assert isinstance(mp5, kp.signals.LazyECPMasterPattern)
        mp5.compute()
        assert np.allclose(mp5.data, mp.data)

    def test_normalize_intensity_lazy_output(self):
        mp = kp.load(ECP_FILE)
        with pytest.raises(
            ValueError, match="`lazy_output=True` requires `inplace=False`"
        ):
            _ = mp.normalize_intensity(lazy_output=True)

        mp2 = mp.normalize_intensity(inplace=False, lazy_output=True)
        assert isinstance(mp2, kp.signals.LazyECPMasterPattern)

        mp3 = mp.as_lazy()
        mp4 = mp3.normalize_intensity(inplace=False, lazy_output=False)
        assert isinstance(mp4, kp.signals.ECPMasterPattern)

    def test_adaptive_histogram_equalization(self):
        mp = kp.load(ECP_FILE)
        mp.rescale_intensity(dtype_out=np.uint8)
        mp.adaptive_histogram_equalization()
        assert all([mp.data.min() >= 0, mp.data.max() <= 255])

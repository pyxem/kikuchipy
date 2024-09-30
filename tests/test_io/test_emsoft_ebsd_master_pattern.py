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

from h5py import File
import numpy as np
import pytest

import kikuchipy as kp


class TestEMsoftEBSDMasterPatternReader:
    @pytest.mark.parametrize(
        "emsoft_ebsd_master_pattern_axes_manager",
        [["energy", "height", "width"]],
        indirect=["emsoft_ebsd_master_pattern_axes_manager"],
    )
    def test_file_reader(
        self,
        emsoft_ebsd_master_pattern_file,
        emsoft_ebsd_master_pattern_metadata,
        emsoft_ebsd_master_pattern_axes_manager,
        assert_dictionary_func,
    ):
        s = kp.load(emsoft_ebsd_master_pattern_file)

        assert s.data.shape == (11, 13, 13)
        assert_dictionary_func(
            s.axes_manager.as_dictionary(), emsoft_ebsd_master_pattern_axes_manager
        )
        assert_dictionary_func(
            s.metadata.as_dictionary(), emsoft_ebsd_master_pattern_metadata
        )

        signal_indx = s.axes_manager.signal_indices_in_array
        assert np.allclose(s.max(axis=signal_indx).data, s.axes_manager["energy"].axis)

    def test_projection_lambert(
        self,
        emsoft_ebsd_master_pattern_file,
        emsoft_ebsd_master_pattern_axes_manager,
        assert_dictionary_func,
    ):
        s = kp.load(
            emsoft_ebsd_master_pattern_file, projection="lambert", hemisphere="both"
        )

        assert s.data.shape == (2, 11, 13, 13)
        assert_dictionary_func(
            s.axes_manager.as_dictionary(), emsoft_ebsd_master_pattern_axes_manager
        )

    @pytest.mark.parametrize("projection", ["stereographic", "lambert"])
    def test_load_lazy(self, emsoft_ebsd_master_pattern_file, projection):
        """The Lambert projection's lower hemisphere is stored chunked."""
        s = kp.load(
            emsoft_ebsd_master_pattern_file,
            projection=projection,
            hemisphere="lower",
            lazy=True,
        )

        assert isinstance(s, kp.signals.LazyEBSDMasterPattern)

        s.compute()

        assert isinstance(s, kp.signals.EBSDMasterPattern)

    @pytest.mark.parametrize(
        "energy, energy_slice, desired_shape, desired_mean_energies",
        [
            (20, slice(10, None), (2, 13, 13), [20]),
            (15, slice(5, 6), (2, 13, 13), [15]),
            ((15, 20), slice(5, None), (2, 6, 13, 13), np.linspace(15, 20, 6)),
            ((19, 20), slice(9, None), (2, 2, 13, 13), np.linspace(19, 20, 2)),
        ],
    )
    def test_load_energy(
        self,
        emsoft_ebsd_master_pattern_file,
        energy,
        energy_slice,
        desired_shape,
        desired_mean_energies,
    ):
        """Ensure desired energy parameters can be passed."""
        s = kp.load(emsoft_ebsd_master_pattern_file, energy=energy, hemisphere="both")
        assert s.data.shape == desired_shape

        s2 = kp.load(
            emsoft_ebsd_master_pattern_file,
            projection="lambert",
            energy=energy,
            hemisphere="upper",
        )
        sig_indx = s2.axes_manager.signal_indices_in_array
        assert np.allclose(s2.nanmean(axis=sig_indx).data, desired_mean_energies)

        with File(emsoft_ebsd_master_pattern_file) as f:
            mp_lambert_upper = f["EMData/EBSDmaster/mLPNH"][:][0][energy_slice]
            assert np.allclose(s2.data, mp_lambert_upper)

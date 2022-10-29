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

from h5py import File
import numpy as np
import pytest

from kikuchipy import load
from kikuchipy.conftest import assert_dictionary
from kikuchipy.signals.ebsd_master_pattern import (
    EBSDMasterPattern,
    LazyEBSDMasterPattern,
)


DIR_PATH = os.path.dirname(__file__)
EMSOFT_FILE = os.path.join(
    DIR_PATH, "../../../data/emsoft_ebsd_master_pattern/master_patterns.h5"
)

METADATA = {
    "General": {"original_filename": "master_patterns.h5", "title": "master_patterns"},
    "Signal": {"binned": False, "signal_type": "EBSDMasterPattern"},
}


AXES_MANAGER = {
    "hemisphere": {
        "name": "hemisphere",
        "scale": 1,
        "offset": 0,
        "size": 2,
        "units": "",
        "navigate": True,
    },
    "energy": {
        "name": "energy",
        "scale": 1,
        "offset": 10.0,
        "size": 11,
        "units": "keV",
        "navigate": True,
    },
    "height": {
        "name": "height",
        "scale": 1,
        "offset": -7.0,
        "size": 13,
        "units": "px",
        "navigate": False,
    },
    "width": {
        "name": "width",
        "scale": 1,
        "offset": -7.0,
        "size": 13,
        "units": "px",
        "navigate": False,
    },
}


def setup_axes_manager(axes=None):
    if axes is None:
        axes = ["hemisphere", "energy", "height", "width"]
    d = {}
    for i, a in enumerate(axes):
        d["axis-" + str(i)] = AXES_MANAGER[a]
    return d


class TestEMsoftEBSDMasterPatternReader:
    def test_file_reader(self):
        s = load(EMSOFT_FILE)

        axes_manager = setup_axes_manager(["energy", "height", "width"])

        assert s.data.shape == (11, 13, 13)
        assert_dictionary(s.axes_manager.as_dictionary(), axes_manager)
        assert_dictionary(s.metadata.as_dictionary(), METADATA)

        signal_indx = s.axes_manager.signal_indices_in_array
        assert np.allclose(s.max(axis=signal_indx).data, s.axes_manager["energy"].axis)

    def test_projection_lambert(self):
        s = load(EMSOFT_FILE, projection="lambert", hemisphere="both")

        assert s.data.shape == (2, 11, 13, 13)
        assert_dictionary(s.axes_manager.as_dictionary(), setup_axes_manager())

    @pytest.mark.parametrize("projection", ["stereographic", "lambert"])
    def test_load_lazy(self, projection):
        """The Lambert projection's lower hemisphere is stored chunked."""
        s = load(EMSOFT_FILE, projection=projection, hemisphere="lower", lazy=True)

        assert isinstance(s, LazyEBSDMasterPattern)

        s.compute()

        assert isinstance(s, EBSDMasterPattern)

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
        self, energy, energy_slice, desired_shape, desired_mean_energies
    ):
        """Ensure desired energy parameters can be passed."""
        s = load(EMSOFT_FILE, energy=energy, hemisphere="both")
        assert s.data.shape == desired_shape

        s2 = load(EMSOFT_FILE, projection="lambert", energy=energy, hemisphere="upper")
        sig_indx = s2.axes_manager.signal_indices_in_array
        assert np.allclose(s2.nanmean(axis=sig_indx).data, desired_mean_energies)

        with File(EMSOFT_FILE) as f:
            mp_lambert_upper = f["EMData/EBSDmaster/mLPNH"][:][0][energy_slice]
            assert np.allclose(s2.data, mp_lambert_upper)

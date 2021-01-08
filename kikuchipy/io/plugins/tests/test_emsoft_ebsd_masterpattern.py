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

import os

from h5py import File
import numpy as np
import pytest

from kikuchipy import load
from kikuchipy.io.plugins.emsoft_ebsd_master_pattern import (
    _check_file_format,
    _get_data_shape_slices,
    _get_datasets,
)
from kikuchipy.signals.ebsd_master_pattern import (
    EBSDMasterPattern,
    LazyEBSDMasterPattern,
)
from kikuchipy.signals.tests.test_ebsd import assert_dictionary


DIR_PATH = os.path.dirname(__file__)
EMSOFT_FILE = os.path.join(
    DIR_PATH, "../../../data/emsoft_ebsd_master_pattern/master_patterns.h5"
)

METADATA = {
    "General": {
        "original_filename": "master_patterns.h5",
        "title": "master_patterns",
    },
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
        assert s.axes_manager.as_dictionary() == axes_manager
        assert_dictionary(s.metadata.as_dictionary(), METADATA)

        signal_indx = s.axes_manager.signal_indices_in_array
        assert np.allclose(
            s.max(axis=signal_indx).data, s.axes_manager["energy"].axis
        )

    def test_projection_lambert(self):
        s = load(EMSOFT_FILE, projection="lambert", hemisphere="both")

        axes_manager = setup_axes_manager()

        assert s.data.shape == (2, 11, 13, 13)
        assert s.axes_manager.as_dictionary() == axes_manager

    def test_check_file_format(self, save_path_hdf5):
        with File(save_path_hdf5, mode="w") as f:
            g1 = f.create_group("EMheader")
            g2 = g1.create_group("EBSDmaster")
            g2.create_dataset(
                "ProgramName",
                data=np.array([b"EMEBSDmasterr.f90"], dtype="S17"),
            )
            with pytest.raises(IOError, match=".* is not in EMsoft's master "):
                _check_file_format(f)

    @pytest.mark.parametrize(
        (
            "npx, energies, energy, expected_shape, expected_slices, "
            "expected_min_max_energy"
        ),
        [
            (
                25,
                np.linspace(1, 10, 10),
                None,
                (10, 51, 51),
                (slice(None, None), slice(None, None), slice(None, None)),
                (1, 10),
            ),
            (
                64,
                np.linspace(10, 20, 11) * 1.5,
                (17.3, 24.7),
                (5, 129, 129),
                (slice(2, 7), slice(None, None), slice(None, None)),
                (18, 24),
            ),
            (
                64,
                np.linspace(10, 20, 11) * 1.5,
                15,
                (1, 129, 129),
                (slice(0, 1), slice(None, None), slice(None, None)),
                (15, 15),
            ),
            (
                64,
                np.linspace(10, 20, 11) * 1.5,
                23,
                (1, 129, 129),
                (slice(5, 6), slice(None, None), slice(None, None)),
                (22.5, 22.5),
            ),
        ],
    )
    def test_get_data_shape_slices(
        self,
        npx,
        energies,
        energy,
        expected_shape,
        expected_slices,
        expected_min_max_energy,
    ):
        data_shape, data_slices = _get_data_shape_slices(
            npx=npx, energies=energies, energy=energy
        )

        assert data_shape == expected_shape
        assert data_slices == expected_slices

        keep_energies = energies[data_slices[0]]
        assert np.allclose(
            (keep_energies.min(), keep_energies.max()), expected_min_max_energy
        )

    @pytest.mark.parametrize("projection", ["stereographic", "lambert"])
    def test_load_lazy(self, projection):
        """The Lambert projection's southern hemisphere is stored
        chunked.
        """
        s = load(
            EMSOFT_FILE, projection=projection, hemisphere="south", lazy=True
        )

        assert isinstance(s, LazyEBSDMasterPattern)

        s.compute()

        assert isinstance(s, EBSDMasterPattern)

    @pytest.mark.parametrize(
        "projection, hemisphere, dataset_names",
        [
            ("stereographic", "North", ["masterSPNH"]),
            ("stereographic", "both", ["masterSPNH", "masterSPSH"]),
            ("lambert", "south", ["mLPSH"]),
            ("Lambert", "BOTH", ["mLPNH", "mLPSH"]),
        ],
    )
    def test_get_datasets(self, projection, hemisphere, dataset_names):
        with File(EMSOFT_FILE, mode="r") as f:
            datasets = _get_datasets(
                data_group=f["EMData/EBSDmaster"],
                projection=projection,
                hemisphere=hemisphere,
            )
            assert [i.name.split("/")[-1] for i in datasets] == dataset_names

    @pytest.mark.parametrize(
        "projection, hemisphere, error_msg",
        [
            ("stereographicl", "north", "'projection' value stereographicl "),
            ("lambert", "east", "'hemisphere' value east "),
        ],
    )
    def test_get_datasets_raises(self, projection, hemisphere, error_msg):
        with File(EMSOFT_FILE, mode="r") as f:
            with pytest.raises(ValueError, match=error_msg):
                _ = _get_datasets(
                    data_group=f["EMData/EBSDmaster"],
                    projection=projection,
                    hemisphere=hemisphere,
                )

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

        s2 = load(
            EMSOFT_FILE, projection="lambert", energy=energy, hemisphere="north"
        )
        sig_indx = s2.axes_manager.signal_indices_in_array
        assert np.allclose(s2.mean(axis=sig_indx).data, desired_mean_energies)

        with File(EMSOFT_FILE, mode="r") as f:
            mp_lambert_north = f["EMData/EBSDmaster/mLPNH"][:][0][energy_slice]
            assert np.allclose(s2.data, mp_lambert_north)

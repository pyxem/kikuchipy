# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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

import h5py
import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.io.plugins.emsoft_ebsd_masterpattern import (
    check_file_format,
    get_data_shape_slices,
    get_datasets,
)

DIR_PATH = os.path.dirname(__file__)
KIKUCHIPY_FILE = os.path.join(DIR_PATH, "../../data/kikuchipy/patterns.h5")
EMSOFT_FILE = os.path.join(
    DIR_PATH, "../../data/emsoft_ebsd_masterpattern/master_patterns.h5"
)


AXES_MANAGER = {
    "y": {
        "name": "y",
        "scale": 1,
        "offset": 0,
        "size": 2,
        "units": "hemisphere",
        "navigate": True,
    },
    "x": {
        "name": "x",
        "scale": 1,
        "offset": 10.0,
        "size": 11,
        "units": "keV",
        "navigate": True,
    },
    "dy": {
        "name": "dy",
        "scale": 1,
        "offset": -7.0,
        "size": 13,
        "units": "px",
        "navigate": False,
    },
    "dx": {
        "name": "dx",
        "scale": 1,
        "offset": -7.0,
        "size": 13,
        "units": "px",
        "navigate": False,
    },
}


def setup_axes_manager(axes=None):
    if axes is None:
        axes = ["y", "x", "dy", "dx"]

    d = {}
    for i, a in enumerate(axes):
        d["axis-" + str(i)] = AXES_MANAGER[a]

    return d


class TestEMsoftEBSDMasterPatternReader:
    def test_file_reader(self):
        s = kp.load(EMSOFT_FILE)

        axes_manager = setup_axes_manager(["x", "dy", "dx"])

        assert s.data.shape == (11, 13, 13)
        assert s.axes_manager.as_dictionary() == axes_manager

    def test_projection_lambert(self):
        s = kp.load(EMSOFT_FILE, projection="lambert", hemisphere="both",)

        axes_manager = setup_axes_manager()

        assert s.data.shape == (2, 11, 13, 13)
        assert s.axes_manager.as_dictionary() == axes_manager

    def test_check_file_format(self, save_path_hdf5):
        with h5py.File(save_path_hdf5, mode="w") as f:
            g1 = f.create_group("EMheader")
            g2 = g1.create_group("EBSDmaster")
            g2.create_dataset(
                "ProgramName",
                data=np.array([b"EMEBSDmasterr.f90"], dtype="S17"),
            )
            with pytest.raises(IOError, match=".* is not in EMsoft's master "):
                _ = check_file_format(f)

    @pytest.mark.parametrize(
        (
            "npx, energies, energy_range, expected_shape, expected_slices, "
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
        ],
    )
    def test_get_data_shape_slices(
        self,
        npx,
        energies,
        energy_range,
        expected_shape,
        expected_slices,
        expected_min_max_energy,
    ):
        data_shape, data_slices = get_data_shape_slices(
            npx=npx, energies=energies, energy_range=energy_range
        )

        assert data_shape == expected_shape
        assert data_slices == expected_slices

        keep_energies = energies[data_slices[0]]
        assert np.allclose(
            (keep_energies.min(), keep_energies.max()), expected_min_max_energy
        )

    def test_wrong_hemisphere_raises(self):
        with h5py.File(EMSOFT_FILE, mode="r") as f:
            hemisphere = "northh"
            with pytest.raises(
                ValueError, match=f"'hemisphere' argument {hemisphere} "
            ):
                _ = get_datasets(
                    data_group=f["EMData/EBSDmaster"],
                    projection="spherical",
                    hemisphere=hemisphere,
                )

    def test_wrong_projection_raises(self):
        with h5py.File(EMSOFT_FILE, mode="r") as f:
            projection = "triangular"
            with pytest.raises(
                ValueError, match=f"'projection' argument {projection}"
            ):
                _ = get_datasets(
                    data_group=f["EMData/EBSDmaster"],
                    projection=projection,
                    hemisphere="north",
                )

    @pytest.mark.parametrize("projection", ["spherical", "lambert"])
    def test_load_lazy(self, projection):
        """The Lambert projection's southern hemisphere is stored
        chunked.

        """

        s = kp.load(
            EMSOFT_FILE, projection=projection, hemisphere="south", lazy=True
        )

        assert isinstance(s, kp.signals.LazyEBSDMasterPattern)

        s.compute()

        assert isinstance(s, kp.signals.EBSDMasterPattern)

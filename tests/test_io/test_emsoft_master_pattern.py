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

from kikuchipy.io.plugins._emsoft_master_pattern import (
    check_file_format,
    get_data_shape_slices,
    get_datasets,
)


class TestEMsoftEBSDMasterPatternReader:
    def test_check_file_format(self, save_path_hdf5):
        with File(save_path_hdf5, mode="w") as f:
            g1 = f.create_group("EMheader")
            g2 = g1.create_group("EBSDmaster")
            g2.create_dataset(
                "ProgramName", data=np.array([b"EMEBSDmasterr.f90"], dtype="S17")
            )
            with pytest.raises(IOError, match=".* is not in EMsoft's master "):
                check_file_format(f, "EBSD")

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
        data_shape, data_slices = get_data_shape_slices(
            npx=npx, energies=energies, energy=energy
        )

        assert data_shape == expected_shape
        assert data_slices == expected_slices

        keep_energies = energies[data_slices[0]]
        assert np.allclose(
            (keep_energies.min(), keep_energies.max()), expected_min_max_energy
        )

    @pytest.mark.parametrize(
        "projection, hemisphere, dataset_names",
        [
            ("stereographic", "Upper", ["masterSPNH"]),
            ("stereographic", "both", ["masterSPNH", "masterSPSH"]),
            ("lambert", "lower", ["mLPSH"]),
            ("Lambert", "BOTH", ["mLPNH", "mLPSH"]),
        ],
    )
    def test_get_datasets(
        self, emsoft_ebsd_master_pattern_file, projection, hemisphere, dataset_names
    ):
        with File(emsoft_ebsd_master_pattern_file) as f:
            datasets = get_datasets(
                data_group=f["EMData/EBSDmaster"],
                projection=projection,
                hemisphere=hemisphere,
            )
            assert [i.name.split("/")[-1] for i in datasets] == dataset_names

    @pytest.mark.parametrize(
        "projection, hemisphere, error_msg",
        [
            ("stereographicl", "upper", "Unknown projection 'stereographicl',"),
            ("lambert", "east", "Unknown hemisphere 'east',"),
        ],
    )
    def test_get_datasets_raises(
        self, emsoft_ebsd_master_pattern_file, projection, hemisphere, error_msg
    ):
        with File(emsoft_ebsd_master_pattern_file) as f:
            with pytest.raises(ValueError, match=error_msg):
                _ = get_datasets(
                    data_group=f["EMData/EBSDmaster"],
                    projection=projection,
                    hemisphere=hemisphere,
                )

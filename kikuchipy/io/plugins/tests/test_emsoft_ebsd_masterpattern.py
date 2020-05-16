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

from h5py import File
import numpy as np
import pytest

from kikuchipy.io._io import load
from kikuchipy.io.plugins.emsoft_ebsd_master_pattern import (
    _check_file_format,
    _crystal_data_2_metadata,
    _dict2dict_via_mapping,
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

# Settings content
METADATA = {
    "General": {
        "original_filename": "master_patterns.h5",
        "title": "master_patterns",
    },
    "Sample": {
        "Phases": {
            "1": {
                "atom_coordinates": {
                    "1": {
                        "atom": 13,
                        "coordinates": np.array([0.1587, 0.6587, 0]),
                        "site_occupation": 1.0,
                        "debye_waller_factor": 0.005,
                    },
                    "2": {
                        "atom": 29,
                        "coordinates": np.array([0, 0, 0.25]),
                        "site_occupation": 1.0,
                        "debye_waller_factor": 0.005,
                    },
                },
                "lattice_constants": np.array(
                    [0.5949, 0.5949, 0.5821, 90, 90, 90]
                ),
                "setting": 1,
                "source": "Su Y.C., Yan J., Lu P.T., Su J.T.: Thermodynamic...",
                "space_group": 140,
            }
        }
    },
    "Signal": {"binned": False, "signal_type": "EBSDMasterPattern"},
    "Simulation": {
        "EBSD_master_pattern": {
            "BSE_simulation": {
                "depth_step": 1.0,
                "energy_step": 1.0,
                "incident_beam_energy": 20.0,
                "max_depth": 100.0,
                "min_beam_energy": 10.0,
                "mode": "CSDA",
                "number_of_electrons": 2000000000,
                "pixels_along_x": 6,
                "sample_tilt": 70,
            },
            "Master_pattern": {
                "Bethe_parameters": {
                    "complete_cutoff": 50.0,
                    "strong_beam_cutoff": 4.0,
                    "weak_beam_cutoff": 8.0,
                },
                "hemisphere": "north",
                "projection": "spherical",
                "smallest_interplanar_spacing": 0.05,
            },
        }
    },
}


AXES_MANAGER = {
    "y": {
        "name": "y",
        "scale": 1,
        "offset": 0,
        "size": 2,
        "units": "hemisphere",
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
        axes = ["y", "energy", "height", "width"]

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

    def test_projection_lambert(self):
        s = load(EMSOFT_FILE, projection="lambert", hemisphere="both",)

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
                _ = _check_file_format(f)

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
        data_shape, data_slices = _get_data_shape_slices(
            npx=npx, energies=energies, energy_range=energy_range
        )

        assert data_shape == expected_shape
        assert data_slices == expected_slices

        keep_energies = energies[data_slices[0]]
        assert np.allclose(
            (keep_energies.min(), keep_energies.max()), expected_min_max_energy
        )

    @pytest.mark.parametrize("projection", ["spherical", "lambert"])
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
            ("spherical", "North", ["masterSPNH"]),
            ("Spherical", "both", ["masterSPNH", "masterSPSH"]),
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
            ("sphericall", "north", "'projection' value sphericall "),
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

    def test_dict2dict_via_mapping(self):
        with File(EMSOFT_FILE, mode="r") as f:
            mc_mapping = [
                ("MCmode", "mode"),
                ("sig", "sample_tilt"),
                ("numsx", "pixels_along_x"),
                ("totnum_el", "number_of_electrons"),
                ("EkeV", "incident_beam_energy"),
                ("Ehistmin", "min_beam_energy"),
                ("Ebinsize", "energy_step"),
                ("depthmax", "max_depth"),
                ("depthstep", "depth_step"),
            ]
            d = _dict2dict_via_mapping(
                dict_in=f["NMLparameters/MCCLNameList"], mapping=mc_mapping,
            )

        actual_keys = list(d.keys())
        actual_keys.sort()
        expected_keys = [j for _, j in mc_mapping]
        expected_keys.sort()
        assert actual_keys == expected_keys

    def test_crystal_data_2_metadata(self):
        group_dict = {
            "Natomtypes": 1,
            "Atomtypes": 13,
            "AtomData": np.array([[0.1587], [0.6587], [0], [1], [0.005]]),
            "CrystalSystem": 2,
            "LatticeParameters": np.array([0.5949, 0.5949, 0.5821, 90, 90, 90]),
            "SpaceGroupNumber": 140,
            "SpaceGroupSetting": 1,
            "Source": "A paper.",
        }
        actual_d = _crystal_data_2_metadata(group_dict)
        desired_d = {
            "atom_coordinates": {
                "1": {
                    "atom": group_dict["Atomtypes"],
                    "coordinates": group_dict["AtomData"][:3, 0],
                    "site_occupation": group_dict["AtomData"][3, 0],
                    "debye_waller_factor": group_dict["AtomData"][4, 0],
                }
            },
            "lattice_constants": group_dict["LatticeParameters"],
            "setting": group_dict["SpaceGroupSetting"],
            "space_group": group_dict["SpaceGroupNumber"],
            "source": group_dict["Source"],
        }

        assert_dictionary(actual_d, desired_d)

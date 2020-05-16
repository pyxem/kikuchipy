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

import dask.array as da
from hyperspy.api import load as hs_load
from hyperspy._signals.signal2d import Signal2D
import numpy as np
import pytest

from kikuchipy import load
from kikuchipy.io.plugins.tests.test_emsoft_ebsd_masterpattern import (
    METADATA,
    setup_axes_manager,
)
from kikuchipy.signals.tests.test_ebsd import assert_dictionary
from kikuchipy.signals.ebsd_master_pattern import (
    EBSDMasterPattern,
    LazyEBSDMasterPattern,
)
from kikuchipy.signals.util._metadata import metadata_nodes


DIR_PATH = os.path.dirname(__file__)
EMSOFT_FILE = os.path.join(
    DIR_PATH, "../../data/emsoft_ebsd_master_pattern/master_patterns.h5"
)


class TestEBSDMasterPatternInit:
    def test_init_no_metadata(self):
        s = EBSDMasterPattern(np.zeros((2, 10, 11, 11)))

        assert s.metadata.has_item("Simulation.EBSD_master_pattern")
        assert s.metadata.has_item("Sample.Phases")

    def test_ebsd_masterpattern_lazy_data_init(self):
        s = EBSDMasterPattern(da.zeros((2, 10, 11, 11)))

        assert isinstance(s, EBSDMasterPattern)
        assert isinstance(s.data, da.Array)

    def test_ebsd_masterpattern_lazy_init(self):
        s = LazyEBSDMasterPattern(da.zeros((2, 10, 11, 11)))

        assert isinstance(s, LazyEBSDMasterPattern)
        assert isinstance(s.data, da.Array)


class TestIO:
    @pytest.mark.parametrize(
        "save_path_hdf5", ["hspy"], indirect=["save_path_hdf5"]
    )
    def test_save_load_hspy(self, save_path_hdf5):
        s = load(EMSOFT_FILE)

        axes_manager = setup_axes_manager(["energy", "height", "width"])

        assert isinstance(s, EBSDMasterPattern)
        assert s.axes_manager.as_dictionary() == axes_manager
        assert_dictionary(s.metadata.as_dictionary(), METADATA)

        s.save(save_path_hdf5)

        s2 = hs_load(save_path_hdf5, signal_type="EBSDMasterPattern")
        assert isinstance(s2, EBSDMasterPattern)
        assert s2.axes_manager.as_dictionary() == axes_manager
        assert_dictionary(s2.metadata.as_dictionary(), METADATA)

        s3 = hs_load(save_path_hdf5)
        assert isinstance(s3, Signal2D)
        s3.set_signal_type("EBSDMasterPattern")
        assert isinstance(s3, EBSDMasterPattern)
        assert s3.axes_manager.as_dictionary() == axes_manager
        assert_dictionary(s.metadata.as_dictionary(), METADATA)


class TestMetadata:
    def test_set_simulation_parameters(self):
        s = EBSDMasterPattern(np.zeros((2, 10, 11, 11)))
        p_desired = {
            "BSE_simulation": {
                "depth_step": 1.0,
                "energy_step": 1.0,
                "incident_beam_energy": 20.0,
                "max_depth": 100.0,
                "min_beam_energy": 10.0,
                "mode": "CSDA",
                "number_of_electrons": 2000000000,
                "pixels_along_x": 5,
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
        p_in = {
            "complete_cutoff": 50.0,
            "depth_step": 1.0,
            "energy_step": 1.0,
            "hemisphere": "north",
            "incident_beam_energy": 20.0,
            "max_depth": 100.0,
            "min_beam_energy": 10.0,
            "mode": "CSDA",
            "number_of_electrons": 2000000000,
            "pixels_along_x": 5,
            "projection": "spherical",
            "sample_tilt": 70,
            "smallest_interplanar_spacing": 0.05,
            "strong_beam_cutoff": 4.0,
            "weak_beam_cutoff": 8.0,
        }
        s.set_simulation_parameters(**p_in)
        ebsd_mp_node = metadata_nodes("ebsd_master_pattern")
        md_dict = s.metadata.get_item(ebsd_mp_node).as_dictionary()
        assert_dictionary(p_desired, md_dict)

    def test_set_phase_parameters(self):
        s = EBSDMasterPattern(np.zeros((2, 10, 11, 11)))
        p = {
            "number": 1,
            "atom_coordinates": {
                "1": {
                    "atom": "Ni",
                    "coordinates": [0, 0, 0],
                    "site_occupation": 1,
                    "debye_waller_factor": 0.0035,
                }
            },
            "formula": "Ni",
            "info": "Some sample info",
            "lattice_constants": [0.35236, 0.35236, 0.35236, 90, 90, 90],
            "laue_group": "m3m",
            "material_name": "Ni",
            "point_group": "432",
            "space_group": 225,
            "setting": 1,
            "source": "Peng",
            "symmetry": 43,
        }
        s.set_phase_parameters(**p)
        md_dict = s.metadata.get_item("Sample.Phases.1").as_dictionary()
        p.pop("number")
        assert_dictionary(p, md_dict)

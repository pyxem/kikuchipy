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
from orix.crystal_map import Phase
import pytest

from kikuchipy import load
from kikuchipy.data import nickel_ebsd_master_pattern_small
from kikuchipy.io.plugins.tests.test_emsoft_ebsd_masterpattern import (
    setup_axes_manager,
    METADATA,
)
from kikuchipy.signals.tests.test_ebsd import assert_dictionary
from kikuchipy.signals.ebsd_master_pattern import (
    EBSDMasterPattern,
    LazyEBSDMasterPattern,
)


DIR_PATH = os.path.dirname(__file__)
EMSOFT_FILE = os.path.join(
    DIR_PATH, "../../data/emsoft_ebsd_master_pattern/master_patterns.h5"
)


class TestEBSDMasterPatternInit:
    def test_init_no_metadata(self):
        s = EBSDMasterPattern(
            np.zeros((2, 10, 11, 11)),
            projection="lambert",
            hemisphere="both",
            phase=Phase("a"),
        )

        assert isinstance(s.phase, Phase)
        assert s.phase.name == "a"
        assert s.projection == "lambert"
        assert s.hemisphere == "both"

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

    @pytest.mark.parametrize(
        "save_path_hdf5", ["hspy"], indirect=["save_path_hdf5"]
    )
    def test_original_metadata_save_load_cycle(self, save_path_hdf5):
        s = nickel_ebsd_master_pattern_small()

        omd_dict_keys = s.original_metadata.as_dictionary().keys()
        desired_keys = [
            "BetheList",
            "EBSDMasterNameList",
            "MCCLNameList",
            "AtomData",
            "Atomtypes",
            "CrystalSystem",
            "LatticeParameters",
            "Natomtypes",
        ]
        assert [k in omd_dict_keys for k in desired_keys]

        s.save(save_path_hdf5)
        s2 = hs_load(save_path_hdf5, signal_type="EBSDMasterPattern")
        assert isinstance(s2, EBSDMasterPattern)

        omd_dict_keys2 = s2.original_metadata.as_dictionary().keys()
        assert [k in omd_dict_keys2 for k in desired_keys]


class TestProperties:
    @pytest.mark.parametrize(
        "projection, hemisphere",
        [("lambert", "north"), ("spherical", "south"), ("lambert", "both")],
    )
    def test_properties(self, projection, hemisphere):
        mp = nickel_ebsd_master_pattern_small(
            projection=projection, hemisphere=hemisphere
        )

        assert mp.projection == projection
        assert mp.hemisphere == hemisphere

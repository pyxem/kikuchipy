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
from orix.vector import Vector3d
from orix.quaternion import Rotation
import pytest


from kikuchipy import load
from kikuchipy.data import nickel_ebsd_master_pattern_small
from kikuchipy.detectors import EBSDDetector
from kikuchipy.io.plugins.tests.test_emsoft_ebsd_masterpattern import (
    setup_axes_manager,
    METADATA,
)
from kikuchipy.signals.tests.test_ebsd import assert_dictionary
from kikuchipy.signals.ebsd_master_pattern import (
    EBSDMasterPattern,
    LazyEBSDMasterPattern,
    _get_direction_cosines,
    _get_lambert_interpolation_parameters,
    _get_patterns_chunk,
    _min_number_of_chunks,
)
from kikuchipy.signals.ebsd import LazyEBSD, EBSD
from kikuchipy.indexing.similarity_metrics import ncc, ndp


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


class TestEBSDCatalogue:
    # Create  detector model
    detector = EBSDDetector(
        shape=(480, 640),
        px_size=50,
        pc=(20, 20, 15000),
        convention="emsoft4",
        tilt=10,
        sample_tilt=70,
    )

    def test_get_direction_cosines(self):
        out = _get_direction_cosines(self.detector)
        assert isinstance(out, Vector3d)

    def test_get_lambert_interpolation_parameters(self):
        dc = _get_direction_cosines(self.detector)
        scl = 500
        npx = 1001
        npy = 1001
        (
            nii,
            nij,
            niip,
            nijp,
            di,
            dj,
            dim,
            djm,
        ) = _get_lambert_interpolation_parameters(dc, scl, npx, npy)

        assert (nii <= niip).all()
        assert (nij <= nijp).all()

        assert (nii < npx).all()
        assert (nij < npy).all()
        assert (niip < npx).all()
        assert (nijp < npx).all()

        assert (nii >= 0).all()
        assert (nij >= 0).all()
        assert (niip >= 0).all()
        assert (nijp >= 0).all()

    def test_get_patterns(self):

        # Ni Test
        EMSOFT_EBSD_FILE = os.path.join(
            DIR_PATH, "../../data/emsoft_ebsd/EBSD_TEST_Ni.h5"
        )
        emsoft_key = load(EMSOFT_EBSD_FILE)
        emsoft_key = emsoft_key.data[0]

        angles = np.array((120, 45, 60))
        r = Rotation.from_euler(np.radians(angles))
        kp_mp = nickel_ebsd_master_pattern_small(
            projection="lambert", hemisphere="both"
        )
        kp_pattern = kp_mp.get_patterns(
            r, self.detector, 20, 100, dtype_out=np.uint8
        )
        kp_pat = kp_pattern.data[0].compute()

        ncc1 = ncc(kp_pat, emsoft_key)
        ndp1 = ndp(kp_pat, emsoft_key)

        assert ncc1 >= 0.935
        assert ndp1 >= 0.935

        detector_shape = self.detector.shape
        r2 = Rotation.from_euler(((0, 0, 0), (1, 1, 1), (2, 2, 2)))
        mp_a = EBSDMasterPattern(np.zeros((2, 10, 11, 11)))
        mp_a.axes_manager[0].name = "energy"
        mp_a.axes_manager[1].name = "y"
        mp_a.projection = "lambert"
        mp_a.phase = Phase("Ni", 225)
        out_a = mp_a.get_patterns(r2, self.detector, 5, 1)

        assert isinstance(out_a, LazyEBSD)
        assert out_a.axes_manager.shape == (
            3,
            detector_shape[1],
            detector_shape[0],
        )

        mp_b = EBSDMasterPattern(np.zeros((10, 11, 11)))
        mp_b.axes_manager[0].name = "energy"
        mp_b.projection = "lambert"
        mp_b.phase = Phase("Ni", 225)
        out_b = mp_b.get_patterns(r2, self.detector, 5, 1)

        assert isinstance(out_b, LazyEBSD)
        assert out_b.axes_manager.shape == (
            3,
            detector_shape[1],
            detector_shape[0],
        )

        mp_c = EBSDMasterPattern(np.zeros((11, 11)))
        mp_c.projection = "lambert"
        mp_c.phase = Phase("Ni", 225)
        out_c = mp_c.get_patterns(r2, self.detector, 5, 1)
        out_c_2 = mp_c.get_patterns(r2, self.detector, 5, 1, compute=True)

        assert isinstance(out_c, LazyEBSD)
        assert isinstance(out_c_2, EBSD)
        assert out_c.axes_manager.shape == (
            3,
            detector_shape[1],
            detector_shape[0],
        )

        mp_c2 = EBSDMasterPattern(np.zeros((11, 11)))
        mp_c2.projection = "lambert"
        mp_c2.phase = Phase("!Ni", 220)
        with pytest.raises(AttributeError):
            mp_c2.get_patterns(r2, self.detector, 5, 1)

        mp_d = EBSDMasterPattern(np.zeros((2, 11, 11)))
        with pytest.raises(NotImplementedError):
            mp_d.get_patterns(r2, self.detector, 5, 1)

        mp_e = EBSDMasterPattern(np.zeros((10, 11, 11)))
        mp_e.axes_manager[0].name = "energy"
        mp_e.projection = "lambert"
        mp_e.phase = Phase("!Ni", 220)
        with pytest.raises(AttributeError):
            mp_e.get_patterns(r2, self.detector, 5, 1)

        # More than one Projection center is currently not supported so it
        # should fail.
        d2 = EBSDDetector(
            shape=(10, 10),
            px_size=50,
            pc=((0, 0, 15000), (0, 0, 15000)),
            convention="emsoft4",
            tilt=0,
            sample_tilt=70,
        )
        with pytest.raises(ValueError):
            mp_c.get_patterns(r2, d2, 5, 1)

        # TODO: Create tests for other structures

    def test_get_patterns_no_chunk(self, detector):
        mp = nickel_ebsd_master_pattern_small(projection="lambert")
        sim = mp.get_patterns(
            rotations=Rotation([1, 0, 0, 0]),
            detector=detector,
            energy=20,
            n_chunk=None,
        )

        assert isinstance(sim, EBSD)
        assert isinstance(sim.data, da.Array)

    def test_get_patterns_chunk(self):
        r = Rotation.from_euler(((0, 0, 0), (1, 1, 1), (2, 2, 2)))
        dc = _get_direction_cosines(self.detector)

        mpn = np.empty((1001, 1001))
        mps = mpn
        npx = 1001
        npy = npx
        out = _get_patterns_chunk(r, dc, mpn, mps, npx, npy, rescale=False)

        assert out.shape == r.shape + dc.shape

    def test_min_number_of_chunks(self):
        n_chunks = _min_number_of_chunks(self.detector.shape, 117000, np.uint8)
        assert n_chunks == 360

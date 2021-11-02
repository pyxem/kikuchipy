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

import dask.array as da
from hyperspy.api import load as hs_load
from hyperspy._signals.signal2d import Signal2D
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy import load
from kikuchipy.data import nickel_ebsd_master_pattern_small
from kikuchipy.io.plugins.tests.test_emsoft_ebsd_masterpattern import (
    setup_axes_manager,
    METADATA,
)
from kikuchipy.signals.tests.test_ebsd import assert_dictionary
from kikuchipy.signals.util._master_pattern import (
    _get_cosine_sine_of_alpha_and_azimuthal,
    _get_direction_cosines_for_multiple_pcs,
    _get_direction_cosines_for_single_pc,
    _get_direction_cosines_for_single_pc_from_detector,
    _get_lambert_interpolation_parameters,
    _get_pixel_from_master_pattern,
    _project_patterns_from_master_pattern,
    _project_single_pattern_from_master_pattern,
)
from kikuchipy.indexing.similarity_metrics import (
    NormalizedCrossCorrelationMetric,
    NormalizedDotProductMetric,
)


DIR_PATH = os.path.dirname(__file__)
EMSOFT_FILE = os.path.join(
    DIR_PATH, "../../data/emsoft_ebsd_master_pattern/master_patterns.h5"
)
EMSOFT_EBSD_FILE = os.path.join(DIR_PATH, "../../data/emsoft_ebsd/EBSD_TEST_Ni.h5")


class TestEBSDMasterPatternInit:
    def test_init_no_metadata(self):
        s = kp.signals.EBSDMasterPattern(
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
        s = kp.signals.EBSDMasterPattern(da.zeros((2, 10, 11, 11)))

        assert isinstance(s, kp.signals.EBSDMasterPattern)
        assert isinstance(s.data, da.Array)

    def test_ebsd_masterpattern_lazy_init(self):
        s = kp.signals.LazyEBSDMasterPattern(da.zeros((2, 10, 11, 11)))

        assert isinstance(s, kp.signals.LazyEBSDMasterPattern)
        assert isinstance(s.data, da.Array)


class TestIO:
    @pytest.mark.parametrize("save_path_hdf5", ["hspy"], indirect=["save_path_hdf5"])
    def test_save_load_hspy(self, save_path_hdf5):
        s = load(EMSOFT_FILE)

        axes_manager = setup_axes_manager(["energy", "height", "width"])

        assert isinstance(s, kp.signals.EBSDMasterPattern)
        assert_dictionary(s.axes_manager.as_dictionary(), axes_manager)
        assert_dictionary(s.metadata.as_dictionary(), METADATA)

        s.save(save_path_hdf5)

        s2 = hs_load(save_path_hdf5, signal_type="EBSDMasterPattern")
        assert isinstance(s2, kp.signals.EBSDMasterPattern)
        assert_dictionary(s2.axes_manager.as_dictionary(), axes_manager)
        assert_dictionary(s2.metadata.as_dictionary(), METADATA)

        s3 = hs_load(save_path_hdf5)
        assert isinstance(s3, Signal2D)
        s3.set_signal_type("EBSDMasterPattern")
        assert isinstance(s3, kp.signals.EBSDMasterPattern)
        assert_dictionary(s3.axes_manager.as_dictionary(), axes_manager)
        assert_dictionary(s.metadata.as_dictionary(), METADATA)

    @pytest.mark.parametrize("save_path_hdf5", ["hspy"], indirect=["save_path_hdf5"])
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
        assert isinstance(s2, kp.signals.EBSDMasterPattern)

        omd_dict_keys2 = s2.original_metadata.as_dictionary().keys()
        assert [k in omd_dict_keys2 for k in desired_keys]


class TestProperties:
    @pytest.mark.parametrize(
        "projection, hemisphere",
        [("lambert", "north"), ("stereographic", "south"), ("lambert", "both")],
    )
    def test_properties(self, projection, hemisphere):
        mp = nickel_ebsd_master_pattern_small(
            projection=projection, hemisphere=hemisphere, lazy=True
        )
        assert mp.projection == projection
        assert mp.hemisphere == hemisphere

        # Deepcopy
        mp2 = mp.deepcopy()

        assert mp2.projection == projection
        mp2.projection = "gnomonic"
        assert mp2.projection != projection

        assert mp2.hemisphere == hemisphere
        mp2.hemisphere = "west"
        assert mp2.hemisphere != hemisphere

        assert mp2.phase.point_group.name == mp.phase.point_group.name
        mp2.phase.space_group = 220
        assert mp2.phase.point_group.name != mp.phase.point_group.name


class TestProjectingPatternsFromLambert:
    detector = kp.detectors.EBSDDetector(
        shape=(480, 640),
        px_size=50,
        pc=(20, 20, 15000),
        convention="emsoft4",
        tilt=10,
        sample_tilt=70,
    )

    def test_get_direction_cosines(self):
        detector = self.detector
        dc = _get_direction_cosines_for_single_pc_from_detector(detector)
        assert dc.shape == detector.shape + (3,)
        assert np.max(dc) <= 1

        dc2 = _get_direction_cosines_for_single_pc.py_func(
            pcx=detector.pcx,
            pcy=detector.pcy,
            pcz=detector.pcz,
            nrows=detector.nrows,
            ncols=detector.ncols,
            tilt=detector.tilt,
            azimuthal=detector.azimuthal,
            sample_tilt=detector.sample_tilt,
        )
        assert np.allclose(dc, dc2)

    def test_get_patterns(self):
        emsoft_key = load(EMSOFT_EBSD_FILE)
        emsoft_key = emsoft_key.data[0]

        r = Rotation.from_euler(np.radians([120, 45, 60]))
        mp1 = nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
        kp_pattern = mp1.get_patterns(
            rotations=r, detector=self.detector, energy=20, dtype_out=emsoft_key.dtype
        )
        kp_pat = kp_pattern.data[0].compute()
        assert kp_pat.dtype == emsoft_key.dtype

        ncc = NormalizedCrossCorrelationMetric(1, 1)
        ncc1 = ncc(kp_pat, emsoft_key).compute()[0][0]
        assert ncc1 >= 0.935

        ndp = NormalizedDotProductMetric(1, 1)
        ndp1 = ndp(kp_pat, emsoft_key).compute()[0][0]
        assert ndp1 >= 0.935

        detector_shape = self.detector.shape
        r2 = Rotation.from_euler(((0, 0, 0), (1, 1, 1), (2, 2, 2)))
        mp2 = kp.signals.EBSDMasterPattern(np.zeros((2, 10, 11, 11)))
        mp2.axes_manager[0].name = "hemisphere"
        mp2.axes_manager[1].name = "energy"
        mp2.projection = "lambert"
        mp2.phase = Phase("Ni", 225)
        out2 = mp2.get_patterns(r2, self.detector, 5)
        assert isinstance(out2, kp.signals.LazyEBSD)
        desired_data_shape = (3,) + detector_shape[::-1]
        assert out2.axes_manager.shape == desired_data_shape

        mp3 = kp.signals.EBSDMasterPattern(np.zeros((10, 11, 11)))
        mp3.axes_manager[0].name = "energy"
        mp3.projection = "lambert"
        mp3.phase = Phase("Ni", 225)
        out3 = mp3.get_patterns(r2, self.detector, 5)
        assert isinstance(out3, kp.signals.LazyEBSD)
        assert out3.axes_manager.shape == desired_data_shape

        mp4 = kp.signals.EBSDMasterPattern(np.zeros((11, 11)))
        mp4.projection = "lambert"
        mp4.phase = Phase("Ni", 225)
        out41 = mp4.get_patterns(r2, self.detector, 5)
        out42 = mp4.get_patterns(r2, self.detector, 5, compute=True)

        assert isinstance(out41, kp.signals.LazyEBSD)
        assert isinstance(out42, kp.signals.EBSD)
        assert out41.axes_manager.shape == desired_data_shape

        mp5 = kp.signals.EBSDMasterPattern(np.zeros((11, 11)))
        mp5.projection = "lambert"
        mp5.phase = Phase("!Ni", 220)
        with pytest.raises(AttributeError):
            _ = mp5.get_patterns(r2, self.detector, 5)

        mp6 = kp.signals.EBSDMasterPattern(np.zeros((2, 11, 11)))
        with pytest.raises(AttributeError, match="Master pattern `phase` attribute"):
            _ = mp6.get_patterns(r2, self.detector, 5)

        mp7 = kp.signals.EBSDMasterPattern(np.zeros((10, 11, 11)))
        mp7.axes_manager[0].name = "energy"
        mp7.projection = "lambert"
        mp7.phase = Phase("!Ni", 220)
        with pytest.raises(AttributeError, match="For point groups without inversion"):
            _ = mp7.get_patterns(r2, self.detector, 5)

        # More than one PC is currently not supported so should fail
        d2 = kp.detectors.EBSDDetector(
            shape=(10, 10),
            px_size=50,
            pc=((0, 0, 15000), (0, 0, 15000)),
            convention="emsoft4",
            sample_tilt=70,
        )
        with pytest.raises(NotImplementedError):
            _ = mp4.get_patterns(r2, d2, 5)

        # TODO: Create tests for other structures

    def test_get_patterns_dtype(self):
        r = Rotation.identity()
        mp = nickel_ebsd_master_pattern_small(projection="lambert")
        dtype_out = np.dtype("float64")
        pattern = mp.get_patterns(
            rotations=r, detector=self.detector, energy=20, dtype_out=dtype_out
        )
        assert pattern.data.dtype == dtype_out

    def test_simulated_patterns_xmap_detector(self):
        mp = nickel_ebsd_master_pattern_small(projection="lambert")
        r = Rotation.from_euler([[0, 0, 0], [0, np.pi / 2, 0]])
        detector = kp.detectors.EBSDDetector(
            shape=(60, 60),
            pc=[0.5, 0.5, 0.5],
            sample_tilt=70,
            convention="tsl",
        )
        s = mp.get_patterns(rotations=r, detector=detector, energy=20)

        assert np.allclose(s.xmap.rotations.to_euler(), r.to_euler())
        assert s.xmap.phases.names == [mp.phase.name]
        assert s.xmap.phases[0].point_group.name == mp.phase.point_group.name

        assert s.detector.shape == detector.shape
        assert np.allclose(s.detector.pc, detector.pc)
        assert s.detector.sample_tilt == detector.sample_tilt

    @pytest.mark.parametrize("nav_shape", [(10, 20), (3, 5), (2, 6)])
    def test_get_patterns_navigation_shape(self, nav_shape):
        mp = nickel_ebsd_master_pattern_small(projection="lambert")
        r = Rotation(np.random.uniform(low=0, high=1, size=nav_shape + (4,)))
        detector = kp.detectors.EBSDDetector(shape=(60, 60))
        sim = mp.get_patterns(rotations=r, detector=detector, energy=20)
        assert sim.axes_manager.navigation_shape[::-1] == nav_shape

    def test_get_patterns_navigation_shape_raises(self):
        mp = nickel_ebsd_master_pattern_small(projection="lambert")
        r = Rotation(np.random.uniform(low=0, high=1, size=(1, 2, 3, 4)))
        detector = kp.detectors.EBSDDetector(shape=(60, 60))
        with pytest.raises(ValueError, match="`rotations` can only have one or two "):
            _ = mp.get_patterns(rotations=r, detector=detector, energy=20)

    def test_detector_azimuthal(self):
        """Test that setting an azimuthal angle of a detector results in
        different patterns.
        """
        det1 = self.detector

        # Looking from the detector toward the sample, the left part of
        # the detector is closer to the sample than the right part
        det2 = det1.deepcopy()
        det2.azimuthal = 10

        # Looking from the detector toward the sample, the right part of
        # the detector is closer to the sample than the left part
        det3 = det1.deepcopy()
        det3.azimuthal = -10

        mp = nickel_ebsd_master_pattern_small(projection="lambert")
        r = Rotation.identity()

        kwargs = dict(rotations=r, energy=20, compute=True, dtype_out=np.uint8)
        sim1 = mp.get_patterns(detector=det1, **kwargs)
        sim2 = mp.get_patterns(detector=det2, **kwargs)
        sim3 = mp.get_patterns(detector=det3, **kwargs)

        assert not np.allclose(sim1.data, sim2.data)
        assert np.allclose(sim2.data.mean(), 43.56, atol=1e-2)
        assert np.allclose(sim3.data.mean(), 43.39, atol=1e-2)

    def test_project_patterns_from_master_pattern(self):
        """Make sure the Numba function is covered."""
        r = Rotation.from_euler(((0, 0, 0), (1, 1, 1), (2, 2, 2)))
        dc = _get_direction_cosines_for_single_pc_from_detector(self.detector)

        npx = npy = 101
        mpn = mps = np.zeros((npy, npx))
        patterns = _project_patterns_from_master_pattern.py_func(
            rotations=r.data,
            direction_cosines=dc,
            master_north=mpn,
            master_south=mps,
            npx=npx,
            npy=npy,
            scale=float((npx - 1) / 2),
            dtype_out=mpn.dtype,
            rescale=False,
            # Aren't used
            out_min=1,
            out_max=2,
        )

        assert patterns.shape == r.shape + dc.shape[:-1]

    @pytest.mark.parametrize(
        "dtype_out, intensity_range", [(np.float32, (0, 1)), (np.uint8, (0, 255))]
    )
    def test_project_single_pattern_from_master_pattern(
        self, dtype_out, intensity_range
    ):
        """Make sure the Numba function is covered."""
        dc = _get_direction_cosines_for_single_pc_from_detector(self.detector)
        npx = npy = 101
        mpn = mps = np.random.random(npy * npx).reshape((npy, npx))

        pattern = _project_single_pattern_from_master_pattern.py_func(
            rotation=np.array([1, 1, 0, 0], dtype=float),
            direction_cosines=dc.reshape((-1, 3)),
            master_north=mpn,
            master_south=mps,
            npx=npx,
            npy=npy,
            scale=1,
            n_pixels=self.detector.size,
            rescale=True,
            out_min=intensity_range[0],
            out_max=intensity_range[1],
            dtype_out=dtype_out,
        )
        assert pattern.shape == (self.detector.size,)
        assert pattern.dtype == dtype_out
        assert np.min(pattern) == intensity_range[0]
        assert np.max(pattern) == intensity_range[1]

    def test_get_lambert_interpolation_parameters(self):
        """Make sure the Numba function is covered."""
        dc = _get_direction_cosines_for_single_pc_from_detector(self.detector)
        npx = npy = 101
        scale = (npx - 1) // 2
        nii, nij, niip, nijp = _get_lambert_interpolation_parameters.py_func(
            v=dc.reshape((-1, 3)), npx=npx, npy=npy, scale=scale
        )[:4]

        assert np.all(nii <= niip)
        assert np.all(nij <= nijp)
        assert np.all(nii < npx)
        assert np.all(nij < npy)
        assert np.all(niip < npx)
        assert np.all(nijp < npx)
        assert np.all(nii >= 0)
        assert np.all(nij >= 0)
        assert np.all(niip >= 0)
        assert np.all(nijp >= 0)

    def test_get_pixel_from_master_pattern(self):
        """Make sure the Numba function is covered."""
        dc = _get_direction_cosines_for_single_pc_from_detector(self.detector)
        npx = npy = 101
        scale = (npx - 1) // 2
        (
            nii,
            nij,
            niip,
            nijp,
            di,
            dj,
            dim,
            djm,
        ) = _get_lambert_interpolation_parameters(
            v=dc.reshape((-1, 3)), npx=npx, npy=npy, scale=scale
        )
        mp = np.ones((npy, npx), dtype=float)
        value = _get_pixel_from_master_pattern.py_func(
            mp, nii[0], nij[0], niip[0], nijp[0], di[0], dj[0], dim[0], djm[0]
        )
        assert value == 1.0

    def test_get_cosine_sine_of_alpha_and_azimuthal(self):
        """Make sure the Numba function is covered."""
        values = _get_cosine_sine_of_alpha_and_azimuthal.py_func(
            sample_tilt=70, tilt=10, azimuthal=5
        )
        assert np.allclose(values, [0.866, 0.5, 0.996, 0.087], atol=1e-3)

    def test_get_direction_cosines_for_multiple_pcs(self):
        """Make sure the Numba function is covered."""
        detector = self.detector
        dc0 = _get_direction_cosines_for_single_pc_from_detector(detector)
        nav_shape = (2, 3)
        detector.pc = np.full(nav_shape + (3,), detector.pc)
        nrows, ncols = detector.shape
        dc = _get_direction_cosines_for_multiple_pcs.py_func(
            pcx=detector.pcx.ravel(),
            pcy=detector.pcy.ravel(),
            pcz=detector.pcz.ravel(),
            nrows=nrows,
            ncols=ncols,
            tilt=detector.tilt,
            azimuthal=detector.azimuthal,
            sample_tilt=detector.sample_tilt,
        )

        assert np.allclose(dc0, dc[0])
        assert dc.shape == (np.prod(nav_shape), nrows, ncols, 3)

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

import dask.array as da
import hyperspy.api as hs
from hyperspy._signals.signal2d import Signal2D
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy.data import nickel_ebsd_master_pattern_small
from kikuchipy.io.plugins.tests.test_emsoft_ebsd_master_pattern import (
    setup_axes_manager,
    METADATA,
)
from kikuchipy.signals.tests.test_ebsd import assert_dictionary
from kikuchipy.signals.util._master_pattern import (
    _get_cosine_sine_of_alpha_and_azimuthal,
    _get_direction_cosines_from_detector,
    _get_direction_cosines_for_fixed_pc,
    _get_direction_cosines_for_varying_pc,
    _get_lambert_interpolation_parameters,
    _get_pixel_from_master_pattern,
    _lambert2vector,
    _project_patterns_from_master_pattern_with_fixed_pc,
    _project_patterns_from_master_pattern_with_varying_pc,
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


class TestEBSDMasterPattern:
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

    def test_ebsd_master_pattern_lazy_data_init(self):
        s = kp.signals.EBSDMasterPattern(da.zeros((2, 10, 11, 11)))

        assert isinstance(s, kp.signals.EBSDMasterPattern)
        assert isinstance(s.data, da.Array)

    def test_ebsd_master_pattern_lazy_init(self):
        s = kp.signals.LazyEBSDMasterPattern(da.zeros((2, 10, 11, 11)))

        assert isinstance(s, kp.signals.LazyEBSDMasterPattern)
        assert isinstance(s.data, da.Array)

    def test_get_master_pattern_arrays_from_energy(self):
        """Get upper and lower hemisphere of master pattern of the
        last energy axis without providing the energy parameter.
        """
        shape = (2, 11, 11)
        data = np.arange(np.prod(shape)).reshape(shape)
        mp = kp.signals.EBSDMasterPattern(
            data,
            axes=[
                {"size": 2, "name": "energy"},
                {"size": 11, "name": "x"},
                {"size": 11, "name": "y"},
            ],
        )
        mp_upper, mp_lower = mp._get_master_pattern_arrays_from_energy()
        assert np.allclose(mp_upper, data[1])
        assert np.allclose(mp_lower, data[1])


class TestIO:
    @pytest.mark.parametrize("save_path_hdf5", ["hspy"], indirect=["save_path_hdf5"])
    def test_save_load_hspy(self, save_path_hdf5):
        s = kp.load(EMSOFT_FILE)

        axes_manager = setup_axes_manager(["energy", "height", "width"])

        assert isinstance(s, kp.signals.EBSDMasterPattern)
        assert_dictionary(s.axes_manager.as_dictionary(), axes_manager)
        assert_dictionary(s.metadata.as_dictionary(), METADATA)

        s.save(save_path_hdf5)

        s2 = hs.load(save_path_hdf5, signal_type="EBSDMasterPattern")
        assert isinstance(s2, kp.signals.EBSDMasterPattern)
        assert_dictionary(s2.axes_manager.as_dictionary(), axes_manager)
        assert_dictionary(s2.metadata.as_dictionary(), METADATA)

        s3 = hs.load(save_path_hdf5)
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
        s2 = hs.load(save_path_hdf5, signal_type="EBSDMasterPattern")
        assert isinstance(s2, kp.signals.EBSDMasterPattern)

        omd_dict_keys2 = s2.original_metadata.as_dictionary().keys()
        assert [k in omd_dict_keys2 for k in desired_keys]


class TestProperties:
    @pytest.mark.parametrize(
        "projection, hemisphere",
        [("lambert", "upper"), ("stereographic", "lower"), ("lambert", "both")],
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


class TestProjectFromLambert:
    detector = kp.detectors.EBSDDetector(
        shape=(480, 640),
        px_size=50,
        pc=(20, 20, 15000),
        convention="emsoft4",
        tilt=10,
    )

    def test_get_direction_cosines(self):
        det = self.detector
        dc = _get_direction_cosines_from_detector(det)
        assert dc.shape == (
            det.size,
            3,
        )
        assert np.max(dc) <= 1

        dc2 = _get_direction_cosines_for_fixed_pc.py_func(
            pcx=det.pcx,
            pcy=det.pcy,
            pcz=det.pcz,
            nrows=det.nrows,
            ncols=det.ncols,
            tilt=det.tilt,
            azimuthal=det.azimuthal,
            sample_tilt=det.sample_tilt,
            mask=np.ones(det.size, dtype=bool),
        )
        assert np.allclose(dc, dc2)

    def test_get_patterns(self):
        emsoft_key = kp.load(EMSOFT_EBSD_FILE)
        emsoft_key = emsoft_key.data[0]

        r = Rotation.from_euler(np.radians([120, 45, 60]))
        mp1 = nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
        kp_pattern = mp1.get_patterns(
            rotations=r, detector=self.detector, dtype_out=emsoft_key.dtype
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
        out2 = mp2.get_patterns(r2, self.detector)
        assert isinstance(out2, kp.signals.LazyEBSD)
        desired_data_shape = (3,) + detector_shape[::-1]
        assert out2.axes_manager.shape == desired_data_shape

        mp3 = kp.signals.EBSDMasterPattern(np.zeros((10, 11, 11)))
        mp3.axes_manager[0].name = "energy"
        mp3.projection = "lambert"
        mp3.phase = Phase("Ni", 225)
        out3 = mp3.get_patterns(r2, self.detector)
        assert isinstance(out3, kp.signals.LazyEBSD)
        assert out3.axes_manager.shape == desired_data_shape

        mp4 = kp.signals.EBSDMasterPattern(np.zeros((11, 11)))
        mp4.projection = "lambert"
        mp4.phase = Phase("Ni", 225)
        out41 = mp4.get_patterns(r2, self.detector)
        out42 = mp4.get_patterns(r2, self.detector, compute=True, show_progressbar=True)

        assert isinstance(out41, kp.signals.LazyEBSD)
        assert isinstance(out42, kp.signals.EBSD)
        assert out41.axes_manager.shape == desired_data_shape

        mp5 = kp.signals.EBSDMasterPattern(np.zeros((11, 11)))
        mp5.projection = "lambert"
        mp5.phase = Phase("!Ni", 220)
        with pytest.raises(AttributeError):
            _ = mp5.get_patterns(r2, self.detector)

        mp6 = kp.signals.EBSDMasterPattern(np.zeros((2, 11, 11)))
        with pytest.raises(AttributeError, match="Master pattern `phase` attribute"):
            _ = mp6.get_patterns(r2, self.detector)

        mp7 = kp.signals.EBSDMasterPattern(np.zeros((10, 11, 11)))
        mp7.axes_manager[0].name = "energy"
        mp7.projection = "lambert"
        mp7.phase = Phase("!Ni", 220)
        with pytest.raises(AttributeError, match="For point groups without inversion"):
            _ = mp7.get_patterns(r2, self.detector)

        # PCs and rotations are not aligned in shape
        d2 = kp.detectors.EBSDDetector(
            shape=(10, 10),
            px_size=50,
            pc=((0, 0, 15000), (0, 0, 15000)),
            convention="emsoft4",
        )
        with pytest.raises(ValueError, match="`detector.navigation_shape` must be "):
            _ = mp4.get_patterns(r2, d2)

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
        """Output signal has the correct navigation shape, and that a
        varying projection center gives different patterns.
        """
        mp = nickel_ebsd_master_pattern_small(projection="lambert")

        # 2D navigation shape, multiple PCs
        rot1 = Rotation.identity(nav_shape)
        det1 = kp.detectors.EBSDDetector(
            shape=(10, 10),
            pc=np.column_stack(
                (
                    np.linspace(0.4, 0.6, rot1.size),
                    np.full(rot1.size, 0.5),
                    np.full(rot1.size, 0.5),
                )
            ).reshape(nav_shape + (3,)),
        )

        sim1 = mp.get_patterns(rotations=rot1, detector=det1)
        assert sim1.axes_manager.navigation_shape[::-1] == nav_shape
        assert not np.allclose(sim1.data[0, 0], sim1.data[0, 1])

        # 1D navigation shape, multiple PCs
        rot2 = rot1.flatten()
        det2 = det1.deepcopy()
        det2.pc = det2.pc.reshape((-1, 3))
        sim2 = mp.get_patterns(rot2, det2)
        assert sim2.axes_manager.navigation_shape == (np.prod(nav_shape),)
        assert np.allclose(sim1.data.reshape((-1,) + det1.shape), sim2.data)

        # 2D navigation shape, single PC
        det2.pc = det2.pc[0]
        sim3 = mp.get_patterns(rot1, det2)
        assert sim3.axes_manager.navigation_shape[::-1] == nav_shape
        assert np.allclose(sim1.data[0, 0], sim3.data[0, 0])

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
        """Cover the Numba functions."""
        r = Rotation.identity((3,))
        det = self.detector.deepcopy()
        det.pc = np.tile(det.pc, (3, 1))
        dc = _get_direction_cosines_from_detector(det)
        dc = dc.reshape((r.size,) + (-1, 3))

        npx = npy = 101
        mpu = mpl = np.zeros((npy, npx))
        kwargs = dict(
            rotations=r.data,
            master_upper=mpu,
            master_lower=mpl,
            npx=npx,
            npy=npy,
            scale=float((npx - 1) / 2),
            dtype_out=mpu.dtype,
            rescale=False,
            # Are not used
            out_min=1,
            out_max=2,
        )

        patterns = _project_patterns_from_master_pattern_with_varying_pc.py_func(
            direction_cosines=dc, **kwargs
        )
        assert patterns.shape == r.shape + (det.size,)

        patterns2 = _project_patterns_from_master_pattern_with_fixed_pc.py_func(
            direction_cosines=dc[0], **kwargs
        )
        assert patterns2.shape == r.shape + (det.size,)

    @pytest.mark.parametrize(
        "dtype_out, intensity_range", [(np.float32, (0, 1)), (np.uint8, (0, 255))]
    )
    def test_project_single_pattern_from_master_pattern(
        self, dtype_out, intensity_range
    ):
        """Make sure the Numba function is covered."""
        dc = _get_direction_cosines_from_detector(self.detector)
        npx = npy = 101
        mpu = mpl = np.random.random(npy * npx).reshape((npy, npx))

        pattern = _project_single_pattern_from_master_pattern.py_func(
            rotation=np.array([1, 1, 0, 0], dtype=float),
            direction_cosines=dc,
            master_upper=mpu,
            master_lower=mpl,
            npx=npx,
            npy=npy,
            scale=1,
            rescale=True,
            out_min=intensity_range[0],
            out_max=intensity_range[1],
            dtype_out=dtype_out,
        )
        assert pattern.shape == (self.detector.size,)
        assert pattern.dtype == dtype_out
        assert np.min(pattern) == intensity_range[0]
        # Windows rescales to 254 instead of 255
        assert np.max(pattern) in [intensity_range[1], intensity_range[1] - 1]

    def test_get_lambert_interpolation_parameters(self):
        """Make sure the Numba function is covered."""
        dc = _get_direction_cosines_from_detector(self.detector)
        npx = npy = 101
        scale = (npx - 1) // 2
        nii, nij, niip, nijp = _get_lambert_interpolation_parameters.py_func(
            v=dc, npx=npx, npy=npy, scale=scale
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
        dc = _get_direction_cosines_from_detector(self.detector)
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
        ) = _get_lambert_interpolation_parameters(v=dc, npx=npx, npy=npy, scale=scale)
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
        det = self.detector
        dc0 = _get_direction_cosines_from_detector(det)
        nav_shape = (2, 3)
        det.pc = np.full(nav_shape + (3,), det.pc)
        nrows, ncols = det.shape
        dc = _get_direction_cosines_for_varying_pc.py_func(
            pcx=det.pcx.ravel(),
            pcy=det.pcy.ravel(),
            pcz=det.pcz.ravel(),
            nrows=nrows,
            ncols=ncols,
            tilt=det.tilt,
            azimuthal=det.azimuthal,
            sample_tilt=det.sample_tilt,
            mask=np.ones(det.size, dtype=bool),
        )

        assert np.allclose(dc0, dc[0])
        assert dc.shape == (np.prod(nav_shape), nrows * ncols, 3)


class TestMasterPatternPlotting:
    @pytest.mark.skipif(not kp._pyvista_installed, reason="PyVista is not installed")
    def test_plot_spherical(self):
        """Returns expected data and raises correct error."""
        import pyvista as pv

        mp = kp.data.nickel_ebsd_master_pattern_small(projection="stereographic")
        pl = mp.plot_spherical(return_figure=True, style="points")
        assert isinstance(pl, pv.Plotter)

        # Number of points equal to points in the master pattern's
        # hemispheres inside equator
        assert pl.mesh.n_points == 251242

        # Actual plot
        mp.plot_spherical(plotter_kwargs=dict(notebook=False))

        # Raise error since only one hemisphere is available and the
        # phase is non-centrosymmetric
        mp.phase.space_group = 185  # P63cm
        with pytest.raises(ValueError):
            mp.plot_spherical()

    @pytest.mark.skipif(kp._pyvista_installed, reason="PyVista is installed")
    def test_plot_spherical_raises(self):  # pragma: no cover
        """Raise ImportError when PyVista is not installed."""
        mp = kp.data.nickel_ebsd_master_pattern_small(projection="stereographic")
        with pytest.raises(ImportError, match="`pyvista` is required"):
            _ = mp.plot_spherical()


class TestAsLambert:
    def test_as_lambert(self, capsys):
        mp_sp = nickel_ebsd_master_pattern_small(projection="stereographic")
        assert mp_sp.projection == "stereographic"
        assert mp_sp.hemisphere == "upper"

        # Upper hemisphere
        mp_lp = mp_sp.as_lambert(show_progressbar=True)
        assert mp_lp.projection == "lambert"
        assert mp_lp.data.shape == mp_sp.data.shape
        assert np.issubdtype(mp_lp.data.dtype, np.float32)
        assert mp_lp.hemisphere == mp_sp.hemisphere
        assert mp_lp.phase.point_group == mp_sp.phase.point_group

        # Warns and raises
        mp_lp_ref = nickel_ebsd_master_pattern_small(projection="lambert")
        with pytest.warns(UserWarning, match="Already in the Lambert projection, "):
            mp_lp_ref2 = mp_lp_ref.as_lambert()
            assert not np.may_share_memory(mp_lp_ref.data, mp_lp_ref2.data)

        mp_sp_lazy = mp_sp.as_lazy()
        with pytest.raises(NotImplementedError, match="Only implemented for non-lazy "):
            _ = mp_sp_lazy.as_lambert()

        # Quite similar to EMsoft's Lambert master pattern
        ncc = NormalizedCrossCorrelationMetric(1, 1)
        assert ncc(mp_lp.data, mp_lp_ref.data).compute() > 0.96

        # "Lower" hemisphere identical to upper
        mp_sp.hemisphere = "lower"
        mp_lp2 = mp_sp.as_lambert(show_progressbar=False)
        out, _ = capsys.readouterr()
        assert not out
        assert mp_lp2.projection == "lambert"
        assert np.allclose(mp_lp.data, mp_lp2.data)

    def test_as_lambert_multiple_energies_hemispheres(self):
        mp_both = nickel_ebsd_master_pattern_small(
            projection="stereographic", hemisphere="both"
        )

        mp_lp_both = mp_both.as_lambert()
        assert mp_lp_both.data.ndim == 3

        # Create a signal with two "energies"
        mp = hs.stack([mp_both, mp_both])
        mp.axes_manager[1].name = "energy"
        mp.hemisphere = mp_both.hemisphere
        mp.projection = mp_both.projection
        mp.phase = mp_both.phase.deepcopy()

        mp_lp_energy = mp.as_lambert()
        assert mp_lp_energy.data.ndim == 4

    def test_lambert2stereographic_numba(self):
        arr = np.linspace(-1, 1, 41, dtype=np.float64)
        x_lambert, y_lambert = np.meshgrid(arr, arr)
        x_lambert_flat = x_lambert.ravel()
        y_lambert_flat = y_lambert.ravel()

        xyz = _lambert2vector.py_func(x_lambert_flat, y_lambert_flat)

        assert xyz.shape == (arr.size**2, 3)
        assert np.all(xyz[:, 2] >= 0)
        assert np.isclose(xyz.max(), 1)
        assert np.isclose(xyz.min(), -1)


class TestIntensityScaling:
    def test_rescale_intensity(self):
        mp = nickel_ebsd_master_pattern_small()
        mp.rescale_intensity(dtype_out=np.float32)
        assert np.allclose([mp.data.min(), mp.data.max()], [-1.0, 1.0])

    def test_normalize_intensity(self):
        mp = nickel_ebsd_master_pattern_small()
        mp.change_dtype("float32")
        mp.normalize_intensity()
        assert np.allclose([mp.data.min(), mp.data.max()], [-1.33, 5.93], atol=1e-2)

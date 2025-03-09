#
# Copyright 2019-2025 the kikuchipy developers
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
#

import dask.array as da
from diffsims.crystallography import ReciprocalLatticeVector
from hyperspy._signals.signal2d import Signal2D
import hyperspy.api as hs
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy._utils._detector_coordinates import (
    convert_coordinates,
    get_coordinate_conversions,
)
from kikuchipy._utils.numba import rotate_vector
from kikuchipy.constants import dependency_version
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_for_fixed_pc,
    _get_direction_cosines_for_varying_pc,
    _get_direction_cosines_from_detector,
    _get_lambert_interpolation_parameters,
    _get_pixel_from_master_pattern,
    _lambert2vector,
    _project_patterns_from_master_pattern_with_fixed_pc,
    _project_patterns_from_master_pattern_with_varying_pc,
    _project_single_pattern_from_master_pattern,
    _vector2lambert,
)


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
    @pytest.mark.parametrize(
        "emsoft_ebsd_master_pattern_axes_manager",
        [["energy", "height", "width"]],
        indirect=["emsoft_ebsd_master_pattern_axes_manager"],
    )
    @pytest.mark.parametrize("save_path_hdf5", ["hspy"], indirect=["save_path_hdf5"])
    def test_save_load_hspy(
        self,
        emsoft_ebsd_master_pattern_file,
        emsoft_ebsd_master_pattern_axes_manager,
        emsoft_ebsd_master_pattern_metadata,
        save_path_hdf5,
        assert_dictionary_func,
    ):
        s = kp.load(emsoft_ebsd_master_pattern_file)

        axman = emsoft_ebsd_master_pattern_axes_manager

        assert isinstance(s, kp.signals.EBSDMasterPattern)
        assert_dictionary_func(s.axes_manager.as_dictionary(), axman)
        assert_dictionary_func(
            s.metadata.as_dictionary(), emsoft_ebsd_master_pattern_metadata
        )

        s.save(save_path_hdf5)

        s2 = hs.load(save_path_hdf5, signal_type="EBSDMasterPattern")
        assert isinstance(s2, kp.signals.EBSDMasterPattern)
        assert_dictionary_func(s2.axes_manager.as_dictionary(), axman)
        assert_dictionary_func(
            s2.metadata.as_dictionary(), emsoft_ebsd_master_pattern_metadata
        )

        s3 = hs.load(save_path_hdf5)
        assert isinstance(s3, Signal2D)
        s3.set_signal_type("EBSDMasterPattern")
        assert isinstance(s3, kp.signals.EBSDMasterPattern)
        assert_dictionary_func(s3.axes_manager.as_dictionary(), axman)
        assert_dictionary_func(
            s.metadata.as_dictionary(), emsoft_ebsd_master_pattern_metadata
        )

    @pytest.mark.parametrize("save_path_hdf5", ["hspy"], indirect=["save_path_hdf5"])
    def test_original_metadata_save_load_cycle(self, save_path_hdf5):
        s = kp.data.nickel_ebsd_master_pattern_small()

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
        mp = kp.data.nickel_ebsd_master_pattern_small(
            projection=projection, hemisphere=hemisphere, lazy=True
        )
        assert mp.projection == projection
        assert mp.hemisphere == hemisphere

        # Deepcopy
        mp2 = mp.deepcopy()

        assert mp2.projection == projection
        with pytest.raises(ValueError, match="Unknown projection 'gnomonic'"):
            mp2.projection = "gnomonic"

        assert mp2.hemisphere == hemisphere
        with pytest.raises(ValueError, match="Unknown hemisphere 'west'"):
            mp2.hemisphere = "west"

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
        """Make sure the Numba function is covered."""
        det = self.detector
        dc = _get_direction_cosines_from_detector(det)
        assert dc.shape == (det.size, 3)
        assert np.max(dc) <= 1

        dc2 = _get_direction_cosines_for_fixed_pc.py_func(
            gnomonic_bounds=det.gnomonic_bounds.squeeze().astype(np.float64),
            pcz=det.pc.squeeze().astype(np.float64)[2],
            nrows=det.nrows,
            ncols=det.ncols,
            om_detector_to_sample=(~det.sample_to_detector).to_matrix().squeeze(),
            signal_mask=np.ones(det.size, dtype=bool),
        )

        dc3 = _get_direction_cosines_for_fixed_pc(
            gnomonic_bounds=det.gnomonic_bounds.squeeze().astype(np.float64),
            pcz=det.pc.squeeze().astype(np.float64)[2],
            nrows=det.nrows,
            ncols=det.ncols,
            om_detector_to_sample=(~det.sample_to_detector).to_matrix().squeeze(),
            signal_mask=np.ones(det.size, dtype=bool),
        )
        assert np.allclose(dc, dc2)
        assert np.allclose(dc2, dc3)

    def test_get_patterns(self, emsoft_ebsd_file):
        emsoft_key = kp.load(emsoft_ebsd_file)
        emsoft_key = emsoft_key.data[0]

        r = Rotation.from_euler(np.radians([120, 45, 60]))
        mp1 = kp.data.nickel_ebsd_master_pattern_small(
            projection="lambert", hemisphere="both"
        )
        kp_pattern = mp1.get_patterns(
            rotations=r, detector=self.detector, dtype_out=emsoft_key.dtype
        )
        kp_pat = kp_pattern.data[0].compute()
        assert kp_pat.dtype == emsoft_key.dtype

        ncc = kp.indexing.NormalizedCrossCorrelationMetric(1, 1)
        ncc1 = ncc(kp_pat, emsoft_key).compute()[0][0]
        assert ncc1 >= 0.935

        ndp = kp.indexing.NormalizedDotProductMetric(1, 1)
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
        mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
        dtype_out = np.dtype("float64")
        pattern = mp.get_patterns(
            rotations=r, detector=self.detector, energy=20, dtype_out=dtype_out
        )
        assert pattern.data.dtype == dtype_out

    def test_simulated_patterns_xmap_detector(self):
        mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
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
        mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")

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
        assert sim1._navigation_shape_rc == nav_shape
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
        assert sim3._navigation_shape_rc == nav_shape
        assert np.allclose(sim1.data[0, 0], sim3.data[0, 0])

    def test_get_patterns_navigation_shape_raises(self):
        mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
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

        mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
        r = Rotation.identity()

        kwargs = dict(rotations=r, energy=20, compute=True, dtype_out=np.uint8)
        sim1 = mp.get_patterns(detector=det1, **kwargs)
        sim2 = mp.get_patterns(detector=det2, **kwargs)
        sim3 = mp.get_patterns(detector=det3, **kwargs)

        assert not np.allclose(sim1.data, sim2.data)
        assert np.allclose(sim2.data.mean(), 43.51, atol=1e-2)
        assert np.allclose(sim3.data.mean(), 43.30, atol=1e-2)

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
        assert np.isclose(value, 1)

    def test_get_direction_cosines_for_multiple_pcs(self):
        """Make sure the Numba function is covered."""
        det = self.detector
        dc0 = _get_direction_cosines_from_detector(det)

        nav_shape = (2, 3)
        det.pc = np.full(nav_shape + (3,), det.pc)
        nrows, ncols = det.shape

        gnomonic_bounds = det.gnomonic_bounds.reshape(
            (np.prod(det.navigation_shape), 4)
        ).astype(np.float64)
        pcz = det.pc_flattened.T.astype(np.float64)[2]

        dc1 = _get_direction_cosines_for_varying_pc.py_func(
            gnomonic_bounds=gnomonic_bounds,
            pcz=pcz,
            nrows=det.nrows,
            ncols=det.ncols,
            om_detector_to_sample=(~det.sample_to_detector).to_matrix().squeeze(),
            signal_mask=np.ones(det.size, dtype=bool),
        )

        dc2 = _get_direction_cosines_for_varying_pc(
            gnomonic_bounds=gnomonic_bounds,
            pcz=pcz,
            nrows=det.nrows,
            ncols=det.ncols,
            om_detector_to_sample=(~det.sample_to_detector).to_matrix().squeeze(),
            signal_mask=np.ones(det.size, dtype=bool),
        )

        assert np.allclose(dc0, dc1[0])
        assert np.allclose(dc1[0], dc2[0])
        assert dc1.shape == (np.prod(nav_shape), nrows * ncols, 3)


class TestMasterPatternPlotting:
    @pytest.mark.skipif(
        dependency_version["pyvista"] is None, reason="PyVista is not installed"
    )
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

    @pytest.mark.skipif(
        dependency_version["pyvista"] is not None, reason="PyVista is installed"
    )
    def test_plot_spherical_raises(self):
        mp = kp.data.nickel_ebsd_master_pattern_small(projection="stereographic")
        with pytest.raises(ImportError):
            _ = mp.plot_spherical()


class TestAsLambert:
    def test_as_lambert(self, capsys):
        mp_sp = kp.data.nickel_ebsd_master_pattern_small(projection="stereographic")
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
        mp_lp_ref = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
        with pytest.warns(UserWarning, match="Already in the Lambert projection, "):
            mp_lp_ref2 = mp_lp_ref.as_lambert()
            assert not np.may_share_memory(mp_lp_ref.data, mp_lp_ref2.data)

        mp_sp_lazy = mp_sp.as_lazy()
        with pytest.raises(NotImplementedError, match="Only implemented for non-lazy "):
            _ = mp_sp_lazy.as_lambert()

        # Quite similar to EMsoft's Lambert master pattern
        ncc = kp.indexing.NormalizedCrossCorrelationMetric(1, 1)
        assert ncc(mp_lp.data, mp_lp_ref.data).compute() > 0.96

        # "Lower" hemisphere identical to upper
        mp_sp.hemisphere = "lower"
        mp_lp2 = mp_sp.as_lambert(show_progressbar=False)
        out, _ = capsys.readouterr()
        assert not out
        assert mp_lp2.projection == "lambert"
        assert np.allclose(mp_lp.data, mp_lp2.data)

    def test_as_lambert_multiple_energies_hemispheres(self):
        mp_both = kp.data.nickel_ebsd_master_pattern_small(
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
        mp = kp.data.nickel_ebsd_master_pattern_small()
        mp.rescale_intensity(dtype_out=np.float32)
        assert np.allclose([mp.data.min(), mp.data.max()], [-1.0, 1.0])

    def test_normalize_intensity(self):
        mp = kp.data.nickel_ebsd_master_pattern_small()
        mp.change_dtype("float32")
        mp.normalize_intensity()
        assert np.allclose([mp.data.min(), mp.data.max()], [-1.33, 5.93], atol=1e-2)

    def test_rescale_intensity_inplace(self):
        mp = kp.data.nickel_ebsd_master_pattern_small()

        # Current signal is unaffected
        mp2 = mp.deepcopy()
        mp3 = mp.normalize_intensity(inplace=False)
        assert isinstance(mp3, kp.signals.EBSDMasterPattern)
        assert np.allclose(mp2.data, mp.data)

        # Operating on current signal gives same result as output
        mp.normalize_intensity()
        assert np.allclose(mp3.data, mp.data)

        # Operating on lazy signal returns lazy signal
        mp4 = mp2.as_lazy()
        mp5 = mp4.normalize_intensity(inplace=False)
        assert isinstance(mp5, kp.signals.LazyEBSDMasterPattern)
        mp5.compute()
        assert np.allclose(mp5.data, mp.data)

    def test_rescale_intensity_lazy_output(self):
        mp = kp.data.nickel_ebsd_master_pattern_small()
        with pytest.raises(
            ValueError, match="'lazy_output=True' requires 'inplace=False'"
        ):
            _ = mp.normalize_intensity(lazy_output=True)

        mp2 = mp.normalize_intensity(inplace=False, lazy_output=True)
        assert isinstance(mp2, kp.signals.LazyEBSDMasterPattern)

        mp3 = mp.as_lazy()
        mp4 = mp3.normalize_intensity(inplace=False, lazy_output=False)
        assert isinstance(mp4, kp.signals.EBSDMasterPattern)

    def test_normalize_intensity_inplace(self):
        mp = kp.data.nickel_ebsd_master_pattern_small()

        # Current signal is unaffected
        mp2 = mp.deepcopy()
        mp3 = mp.normalize_intensity(inplace=False)
        assert isinstance(mp3, kp.signals.EBSDMasterPattern)
        assert np.allclose(mp2.data, mp.data)

        # Operating on current signal gives same result as output
        mp.normalize_intensity()
        assert np.allclose(mp3.data, mp.data)

        # Operating on lazy signal returns lazy signal
        mp4 = mp2.as_lazy()
        mp5 = mp4.normalize_intensity(inplace=False)
        assert isinstance(mp5, kp.signals.LazyEBSDMasterPattern)
        mp5.compute()
        assert np.allclose(mp5.data, mp.data)

    def test_normalize_intensity_lazy_output(self):
        mp = kp.data.nickel_ebsd_master_pattern_small()
        with pytest.raises(
            ValueError, match="'lazy_output=True' requires 'inplace=False'"
        ):
            _ = mp.normalize_intensity(lazy_output=True)

        mp2 = mp.normalize_intensity(inplace=False, lazy_output=True)
        assert isinstance(mp2, kp.signals.LazyEBSDMasterPattern)

        mp3 = mp.as_lazy()
        mp4 = mp3.normalize_intensity(inplace=False, lazy_output=False)
        assert isinstance(mp4, kp.signals.EBSDMasterPattern)

    @pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
    def test_adaptive_histogram_equalization(self):
        mp_sp = kp.data.nickel_ebsd_master_pattern_small()

        # Float warns
        mp_sp.change_dtype(np.float32)
        mp_sp2 = mp_sp.rescale_intensity(inplace=False)
        with pytest.warns(UserWarning, match="Equalization of signals with floating "):
            mp_sp2.adaptive_histogram_equalization()

        # NaN warns
        mp_sp.data[mp_sp.data == 0] = np.nan
        with pytest.warns(UserWarning, match="Equalization of signals with NaN "):
            mp_sp.adaptive_histogram_equalization()

        # Spreads intensities within data range
        mp_lp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
        mp_lp2 = mp_lp.adaptive_histogram_equalization(inplace=False)
        assert all([mp_lp2.data.min() >= 0, mp_lp2.data.max() <= 255])
        assert abs(np.unique(mp_lp2.data).size - 255) < 2


class TestLambertProjection:
    def test_vector2xy(self):
        """Works for numpy arrays."""
        xyz = np.array(
            [
                [0, 0, 1],
                [0, 1, 0],
                [2, 0, 0],
                [0, 0, -3],
                [0, 0, -1],
                [0, -1, 0],
                [-2, 0, 0],
                [0, 0, 3],
            ],
            dtype=np.float64,
        )
        lambert_xy = [
            [0, 0],
            [0, np.sqrt(np.pi / 2)],
            [np.sqrt(np.pi / 2), 0],
            [0, 0],
            [0, 0],
            [0, -np.sqrt(np.pi / 2)],
            [-np.sqrt(np.pi / 2), 0],
            [0, 0],
        ]
        assert np.allclose(_vector2lambert.py_func(xyz), lambert_xy)
        assert np.allclose(_vector2lambert(xyz), lambert_xy)


class TestFitPatternDetectorOrientation:
    """
    Test the fit between an EBSD pattern generated
    from a master pattern and an associated
    GeometricalKikuchiPatternSimulation for different
    detector orientations.
    """

    detector = kp.detectors.EBSDDetector(
        shape=(480, 640), px_size=50, pc=(20, 20, 15000), convention="emsoft4"
    )

    phase = kp.data.nickel_ebsd_master_pattern_small().phase

    hkl = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1)]

    ref = ReciprocalLatticeVector(phase=phase, hkl=hkl)
    ref = ref.symmetrise()

    simulator = kp.simulations.KikuchiPatternSimulator(ref)

    rotations = Rotation.from_euler([23, 14, 5], degrees=True)

    # Transformation from CSs to cartesian crystal reference frame CSc
    u_o = rotations.to_matrix().squeeze()

    # Transformation from CSc to direct crystal reference frame CSk
    u_a = phase.structure.lattice.base

    def setup_detector_sim_and_u_os(self, tilt, azimuthal, twist):
        det = self.detector.deepcopy()
        det.tilt = tilt
        det.azimuthal = azimuthal
        det.twist = twist

        sim_lines = self.simulator.on_detector(det, self.rotations)

        u_os = np.matmul(self.u_o, det.sample_to_detector.to_matrix().squeeze())

        return det, sim_lines, u_os

    def setup_za_and_cds(self, zone_axes, sim_lines, det):
        """Find the indices of the zone_axes in the
        GeometricalKikuchiPatternSimulation (sim_lines).
        Find the gnomonic coordinates of these zone axes.
        Then obtain convert these into pixel coordinates
        on the detector and round to the nearest pixel.
        Return the indices of the zone axes, the
        conversion dict from pix to gn and back, and
        the coordinates of the nearest detector pixels
        to the three zone axes.
        """
        za_idx = [
            index_row_in_array(sim_lines._zone_axes.vector.uvw, i) for i in zone_axes
        ]

        x_gn = sim_lines._zone_axes.x_gnomonic[0, za_idx]
        y_gn = sim_lines._zone_axes.y_gnomonic[0, za_idx]
        cds_gn = np.stack((x_gn, y_gn), axis=1)

        conversions = get_coordinate_conversions(det.gnomonic_bounds, det.bounds)
        cds_px = convert_coordinates(cds_gn, "gn_to_pix", conversions)

        cds_px_int = np.squeeze(np.around(cds_px, decimals=0).astype(int))

        return za_idx, conversions, cds_px_int

    def get_a_angles(self, sim_lines, za_idx, u_os):
        """Check that the GeometricalKikuchiPatternSimulation
        is self-consistent. We do this by taking the vectors
        in the detector reference frame of 3 zone axes in
        the simulation, transforming these vectors from CSd
        to the cartesian crystal reference frame, CSc, and
        measuring the angle between these and vectors
        transformed from the uvw indices of the zone axes
        into the cartesian crystal reference frame
        i.e. CSk to CSc. These angles (a_ang) should all be
        zero if the GeometricalKikuchiPatternSimulation is
        self-consistent. This is tested later.
        The a_ang are returned.
        """
        # CSk to CSc:
        uvw = sim_lines._zone_axes.vector.uvw[za_idx]
        d_vec = np.matmul(uvw, self.u_a)
        N1 = np.sqrt(np.sum(np.square(d_vec), axis=1))
        d_vec_n = d_vec / np.expand_dims(N1, 1)

        # CSd to CSc:
        za_vec = sim_lines._zone_axes.vector_detector[0, za_idx].data
        za_vec_trans = np.matmul(za_vec, np.linalg.inv(u_os)).squeeze()
        N2 = np.sqrt(np.sum(np.square(za_vec_trans), axis=1))
        za_vec_trans_n = za_vec_trans / np.expand_dims(N2, 1)

        # angles between the two sets of vectors:
        a_ang = np.array(
            [
                np.rad2deg(
                    np.arccos(
                        np.around(
                            np.dot(za_vec_trans_n[i, :], d_vec_n[i, :]), decimals=8
                        )
                    )
                )
                for i in range(3)
            ]
        )

        return a_ang, d_vec_n

    def get_d_ang(self, cds_px_int, conversions, u_os, d_vec_n, det):
        """Find the gnomonic coordinates of the nearest
        pixel centre to each of the 3 zone axes. Turn them
        into vectors and transform them from CSd to CSc using
        the same transformation as the
        GeometricalKikuchiPatternSimulation. Then
        calculate the angles (d_ang) between these vectors
        and the vectors representing the zone axes in CSc but
        calculated from CSk to CSc (d_vec_n).
        """
        # Here we add 0.5 because pixel centres are used by the direction
        # cosines method and we need to take the same coords for this
        # alternative approach.
        cds_gn_int = np.squeeze(
            convert_coordinates(cds_px_int + 0.5, "pix_to_gn", conversions)
        )

        vecs = np.hstack(
            (
                cds_gn_int * det.pcz,
                np.repeat(np.atleast_2d(det.pcz), cds_gn_int.shape[0], axis=0),
            )
        )
        N3 = np.sqrt(np.sum(np.square(vecs), axis=1))
        vecs_n = vecs / np.expand_dims(N3, 1)

        # CSd to CSc:
        dddd = np.matmul(vecs_n, np.linalg.inv(u_os)).squeeze()

        d_ang = np.array(
            [
                np.rad2deg(
                    np.arccos(np.around(np.dot(dddd[i, :], d_vec_n[i, :]), decimals=8))
                )
                for i in range(3)
            ]
        )

        return d_ang, dddd

    def get_r_ang_and_n_ang(self, cds_px_int, det, d_vec_n, dddd):
        """Calculate the direction cosines of all the
        detector pixels then rotate them to account
        for the crystal orientation. The resulting
        vectors are in CSc. Select the vectors
        corresponding to the pixel centres representing the 3
        zone axes. Then calculate the angles (r_ang) between
        these vectors and the vectors representing the
        zone axes in CSc but calculated from CSk to CSc
        (d_vec_n). Finally calculate the angles (n_ang) between
        the two sets of vectors representing the centres of the
        nearest pixels to the zone axes (one set is from the direction
        cosines of the detecotr, the other is from the
        GeometricalKikuchiPatternSimulation). The angles are
        zero if the transformations used for the direction
        cosines and for the GeometricalKikuchiPatternSimulation
        are the same (this is tested later).
        """
        # swap columns for i,j array indexing:
        cds_px_int_ij = cds_px_int[:, ::-1]

        # all detector pixels:
        r_g_array = _get_direction_cosines_from_detector(det)
        r_g_array_rot = rotate_vector(self.rotations.data[0], r_g_array)
        rgarrrot_reshaped = r_g_array_rot.reshape((*self.detector.shape, 3))

        # select vectors corresponding to the nearest pixels to the chosen zone axes
        rgrar_vec = rgarrrot_reshaped[cds_px_int_ij[:, 0], cds_px_int_ij[:, 1]]

        r_ang = np.array(
            [
                np.rad2deg(
                    np.arccos(
                        np.around(np.dot(d_vec_n[i, :], rgrar_vec[i, :]), decimals=8)
                    )
                )
                for i in range(3)
            ]
        )

        n_ang = np.array(
            [
                np.rad2deg(
                    np.arccos(
                        np.around(np.dot(dddd[i, :], rgrar_vec[i, :]), decimals=8)
                    )
                )
                for i in range(3)
            ]
        )

        return r_ang, n_ang

    def calculate_fit(self, tilt, azimuthal, twist, zone_axes):
        """
         Calculates four sets of angles with which the fit
         between the EBSD pattern simulated from a master
         pattern, and the GeometricalKikuchiPatternSimulation
         generated using the same parameters, can be evaluated.
         The function can be tested with different values of
         the detector tilt, azimuthal and euler_2 angles and
         appropriate zone axes (indices uvw) which appear on
         the pattern under those conditions.

         The approach has several steps:

         1. Check that the GeometricalKikuchiPatternSimulation
         is self-consistent. We do this by taking the vectors
         in the detector reference frame of 3 zone axes in
         the simulation, transforming these vectors from CSd
         to the cartesian crystal reference frame, CSc, and
         measuring the angle between these and vectors
         transformed from the uvw indices of the zone axes
         into the cartesian crystal reference frame
         i.e. CSk to CSc. These angles (a_ang) should all be
         zero if the GeometricalKikuchiPatternSimulation is
         self-consistent.

         2. Find the gnomonic coordinates of the nearest
         pixel centre to the 3 zone axes. This enables us
         to check the vectors corresponding to the detector
         pixels. We do two things with these coordinates.
         a) turn them into vectors and transform them
         from CSd to CSc using the same transformation as
         the GeometricalKikuchiPatternSimulation. Then
         calculate the angles (d_ang) between these vectors
         and the vectors representing the zone axes in CSc.
         b) calculate the direction cosines of all the
         detector pixels then rotate them to account
         for the crystal orientation. The resulting
         vectors are in CSc but calculated using
         different functions. Select the vectors
         corresponding to the pixels representing the 3
         zone axes. Then calculate the angles (r_ang) between
         these vectors and the vectors representing the
         zone axes in CSc.
         These angles should be the same for a) and b),
         meaning that the angle between the zone axes
         and the centre of the nearest pixel is the
         same for both transformation routes.

         3. Finally calculate the angles (n_ang) between the two
         sets of vectors representing the centres of the
         nearest pixels to the zone axes. The angles are
         zero if the transformations used for the direction
         cosines and for the GeometricalKikuchiPatternSimulation
         are the same.


         Parameters
         ----------
         tilt : Float
             The detector tilt angle in degrees (i.e. detector.tilt).
             Detector Euler angle PHI (EBSDDetector.euler[1]) == 90 + tilt
         azimuthal : Float
             The detector azimuthal angle in degrees (i.e. detector.azimuthal).
             Detector Euler angle phi1 (EBSDDetector.euler[0]) == azimuthal
        twist : Float
             The detector twist angle (EBSDDetector.euler[2]) in deg.
         zone_axes : List or np.ndarray
             List/array containing three lists, each containing a set
             of uvw indices describing a zone axis on the pattern,
             e.g. [[0,0,1], [1,1,0], [1,2,3]].

         Returns
         -------
         a_ang : np.ndarray
             The angles in degrees, between vectors in CSc calculated
             1) from uvw indeces in CSk and 2) from vectors in CSd.
             These should all be zero for self-consistency of
             the GeometricalKikuchiPatternSimulation.
         d_ang : np.ndarray
             The angles in degrees, between vectors representing
             3 zone axes on the pattern and vectors representing
             the nearest pixel centres to the same zone axes.
             Both sets of vectors are transformed into CSc.
             The transformation for both sets was the one used
             by the GeometricalKikuchiPatternSimulation.
         r_ang : np.ndarray
             The angles in degrees, between vectors representing
             3 zone axes on the pattern and vectors representing
             the nearest pixel centres to the same zone axes but
             this time, the pixel centre vectors use the
             transformation of the direction cosines of the
             detector pixels.
         n_ang : np.ndarray
             The angles in degrees between the two sets of vectors
             representing the centres of the nearest pixels to 3
             zone axes on the pattern. Both sets of vectors are
             in CSc. The transformation for one set was the one
             used by the GeometricalKikuchiPatternSimulation,
             and the one used by the other set was for the
             direction cosines of the detector. These angles
             should all be zero if the pattern and simulation
             match.
        """
        det, sim_lines, u_os = self.setup_detector_sim_and_u_os(tilt, azimuthal, twist)
        za_idx, conversions, cds_px_int = self.setup_za_and_cds(
            zone_axes, sim_lines, det
        )
        a_ang, d_vec_n = self.get_a_angles(sim_lines, za_idx, u_os)
        d_ang, dddd = self.get_d_ang(cds_px_int, conversions, u_os, d_vec_n, det)
        r_ang, n_ang = self.get_r_ang_and_n_ang(cds_px_int, det, d_vec_n, dddd)

        return a_ang, d_ang, r_ang, n_ang

    @pytest.mark.parametrize(
        "tilt, azimuthal, twist, zone_axes",
        [
            (0.0, 0.0, 0.0, [[1, 0, 1], [0, 0, 1], [1, 1, 2]]),
            (0.0, 0.0, 1.2, [[1, 0, 1], [0, 0, 1], [1, 1, 2]]),
            (40.0, 0.0, 0.0, [[1, 0, 1], [1, 0, 0], [1, -2, 1]]),
            (40.0, 0.0, 1.2, [[1, 0, 1], [1, 0, 0], [1, -2, 1]]),
            (0.0, 40.0, 0.0, [[1, 0, 1], [1, 1, 0], [1, 2, 1]]),
            (0.0, 40.0, 1.2, [[1, 0, 1], [1, 1, 0], [1, 2, 1]]),
            (40.0, 40.0, 0.0, [[1, 0, 1], [1, 0, 0], [3, 1, 0]]),
            (40.0, 40.0, 1.2, [[1, 0, 1], [1, 0, 0], [3, 1, 0]]),
        ],
    )
    def test_fit_detector_orientation(self, tilt, azimuthal, twist, zone_axes):
        """
        Check that the EBSD pattern simulated from a master
        pattern and the associated
        GeometricalKikuchiPatternSimulation match perfectly,
        for various detector orientations.

        4 sets of angles are returned by self.calculate_fit().
        See the doctstring of that function for details.

        Here we assert that the first set of angles are all
        zero, that the second and third sets are equal, and
        that the fourth set are all zero. If these conditions
        are all met, the GeometricalKikuchiPatternSimulation
        should match the EBSD pattern simulated from a
        master pattern perfectly for the given detector
        orientations.


        Parameters
        ----------
        tilt : Float
            The detector tilt angle in degrees (i.e. detector.tilt).
            Detector Euler angle PHI (EBSDDetector.euler[1]) == 90 + tilt
        azimuthal : Float
            The detector azimuthal angle in degrees (i.e. detector.azimuthal).
            Detector Euler angle phi1 (EBSDDetector.euler[0]) == azimuthal
        twist : Float
            The detector twist angle (EBSDDetector.euler[2]) in deg.
        zone_axes : List or np.ndarray
            List/array containing three lists, each containing a set
            of uvw indices describing a zone axis on the pattern,
            e.g. [[0,0,1], [1,1,0], [1,2,3]].

        Returns
        -------
        None.

        """
        angles = self.calculate_fit(tilt, azimuthal, twist, zone_axes)

        assert np.allclose(angles[0], 0.0)
        assert np.allclose(angles[1], angles[2])
        assert np.allclose(angles[3], 0.0)


def index_row_in_array(myarray, myrow):
    """Check if the row "myrow" is present in the array "myarray".
    If it is, return an int containing the row index of the first
    occurrence. If the row is not present, return None.
    """
    loc = np.where((myarray == myrow).all(-1))[0]
    if len(loc) > 0:
        return loc[0]
    return None

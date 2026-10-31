#
# Copyright 2019-2026 the kikuchipy developers
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
import numpy as np
import orix.quaternion as oqu
import pytest

import kikuchipy as kp
from kikuchipy._kikuchi_sphere._detector_to_stereo import (
    _accumulate_intensities,
    _detector_to_stereo,
    _get_crystal_to_detector_matrices,
    _get_unit_sphere_vectors,
    _normalize_accumulator,
    _project_detector_to_sphere,
)
from kikuchipy.indexing.similarity_metrics._normalized_cross_correlation import (
    _ncc_single_patterns_2d_ref_normalized,
    _zero_mean_normalize_pattern_numpy,
)


class TestGetUnitSphereVectors:
    def test_unit_sphere_vectors(self):
        size = 11
        v_upper, v_lower, inside_equator = _get_unit_sphere_vectors(size)
        assert v_upper.shape == v_lower.shape == (size * size, 3)
        assert inside_equator.shape == (size * size,)
        assert v_upper.flags.c_contiguous
        assert v_lower.flags.c_contiguous
        assert np.allclose(np.linalg.norm(v_upper, axis=1), 1)

        # Hemispheres are mirrored about the equator
        assert np.allclose(v_upper[:, :2], v_lower[:, :2])
        assert np.allclose(v_upper[:, 2], -v_lower[:, 2])
        assert np.all(v_upper[inside_equator, 2] >= 0)
        assert np.all(v_upper[~inside_equator, 2] < 0)

        # Center pixel is the north pole, top row has positive y
        assert np.allclose(v_upper[size * size // 2], [0, 0, 1])
        assert np.all(v_upper.reshape(size, size, 3)[0, :, 1] > 0)

        # Corners are outside the disk, center is inside
        mask = inside_equator.reshape(size, size)
        assert not mask[0, 0]
        assert mask[size // 2, size // 2]

    def test_include_outside_equator(self):
        n = 11
        v_upper1, v_lower1, inside_disk1 = _get_unit_sphere_vectors(n)
        v_upper2, v_lower2, inside_disk2 = _get_unit_sphere_vectors(
            n, exclude_outside_equator=False
        )
        assert np.allclose(v_upper1, v_upper2)
        assert np.allclose(v_lower1, v_lower2)
        assert not inside_disk1.all()
        assert inside_disk2.all()


class TestGetCrystalToDetectorMatrices:
    def test_matrices(self, detector):
        rot = oqu.Rotation.from_euler(np.deg2rad([[10, 20, 30], [50, 40, 10]]))
        rotm_c2d = _get_crystal_to_detector_matrices(rot, detector)
        assert rotm_c2d.shape == (2, 1, 3, 3)
        assert rotm_c2d.flags.c_contiguous
        for i in range(2):
            rotm_c2d_i = (detector.sample_to_detector * ~rot[i]).to_matrix()
            assert np.allclose(rotm_c2d[i, 0], rotm_c2d_i.squeeze())

    def test_matrices_symmetrized(self, detector):
        pg = kp.data.nickel_ebsd_master_pattern_small().phase.point_group
        ori = oqu.Orientation.from_euler(
            np.deg2rad([[10, 20, 30], [50, 40, 10]]), symmetry=pg
        )
        rotm_c2d = _get_crystal_to_detector_matrices(ori, detector, symmetrize=True)
        assert rotm_c2d.shape == (2, pg.proper_subgroup.size, 3, 3)

        # Same as applying each symmetry operator in the crystal
        rotm_s2d = detector.sample_to_detector.to_matrix().squeeze()
        for i in range(2):
            for j, sym_op in enumerate(pg.proper_subgroup):
                rotm_c2s_eq = (~(sym_op * ori[i])).to_matrix().squeeze()
                assert np.allclose(rotm_c2d[i, j], rotm_s2d @ rotm_c2s_eq)


class TestAccumulateAndNormalize:
    def test_accumulate_normalize(self):
        accumulator = np.zeros((2, 3), dtype=np.float64)
        counts = np.zeros((2, 3), dtype=np.int32)
        partial1 = np.array([[1, np.nan, 2], [np.nan, np.nan, 4]])
        partial2 = np.array([[3, np.nan, np.nan], [np.nan, 5, 6]])
        _accumulate_intensities(accumulator, counts, partial1)
        _accumulate_intensities(accumulator, counts, partial2)
        assert np.allclose(counts, [[2, 0, 1], [0, 1, 2]])

        sphere = _normalize_accumulator(accumulator, counts)
        assert np.allclose(sphere, [[2, np.nan, 2], [np.nan, 5, 5]], equal_nan=True)

    def test_accumulate_normalize_py_func(self):
        """Ensure coverage of Numba decorated functions."""
        accumulator = np.zeros((2, 2), dtype=np.float64)
        counts = np.zeros((2, 2), dtype=np.int32)
        partial = np.array([[1, np.nan], [np.nan, 4]])
        _accumulate_intensities.py_func(accumulator, counts, partial)
        sphere = _normalize_accumulator.py_func(accumulator, counts)
        assert np.allclose(sphere, partial, equal_nan=True)


class TestProjectDetectorToSphere:
    def test_project_py_func(self, detector):
        """Ensure coverage of Numba decorated function and that the
        compiled and Python functions give the same result.
        """
        n = 21
        v_upper, _, inside_disk = _get_unit_sphere_vectors(n)
        rot = oqu.Rotation.from_axes_angles([1, 0, 0], -np.pi / 2)
        rotm_c2d = _get_crystal_to_detector_matrices(rot, detector)[0, 0]
        gn_x0, gn_y1, x_scale, y_scale = detector._get_gnomonic_parameters()[0]
        pattern = np.random.default_rng(42).random(detector.shape)
        args = (
            v_upper,
            inside_disk,
            rotm_c2d,
            pattern,
            gn_x0,
            gn_y1,
            x_scale,
            y_scale,
            detector.nrows,
            detector.ncols,
        )
        out1 = _project_detector_to_sphere(*args)
        out2 = _project_detector_to_sphere.py_func(*args)
        assert np.isfinite(out1).any()
        assert np.allclose(out1, out2, equal_nan=True)


class TestDetectorToStereo:
    def test_round_trip_master_pattern(self, detector):
        """Projecting a pattern simulated from a master pattern back to
        the sphere reproduces the master pattern.
        """
        mp_lambert = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
        mp_stereo = kp.data.nickel_ebsd_master_pattern_small(projection="stereographic")
        n = mp_stereo.axes_manager.signal_shape[0]

        rot = oqu.Rotation.from_euler(np.deg2rad([10, 20, 30]))
        pattern = mp_lambert.get_patterns(
            rot, detector, compute=True, show_progressbar=False
        ).data

        sphere = _detector_to_stereo(
            detector, pattern.reshape((1,) + detector.shape), rot, size=n
        )
        assert sphere.shape == (2, n, n)

        upper = sphere[0]
        is_valid = np.isfinite(upper)
        assert is_valid.sum() > 1000
        assert np.isnan(sphere[1]).all()

        signal_mask = ~is_valid
        ref = _zero_mean_normalize_pattern_numpy(mp_stereo.data, signal_mask)
        ncc = _ncc_single_patterns_2d_ref_normalized(upper, ref, signal_mask)
        assert ncc > 0.9

        # Wrong orientation gives a poor match
        rot2 = oqu.Rotation.from_euler(np.deg2rad([40, 20, 30]))
        sphere2 = _detector_to_stereo(
            detector, pattern.reshape((1,) + detector.shape), rot2, size=n
        )
        signal_mask2 = ~np.isfinite(sphere2[0])
        ref2 = _zero_mean_normalize_pattern_numpy(mp_stereo.data, signal_mask2)
        ncc2 = _ncc_single_patterns_2d_ref_normalized(sphere2[0], ref2, signal_mask2)
        assert ncc2 < 0.5

    def test_mean_of_patterns(self, detector):
        """Combining patterns gives the per-pixel mean over contributing
        patterns.
        """
        n = 51
        rot = oqu.Rotation.from_euler(np.deg2rad([[10, 20, 30], [15, 20, 30]]))
        patterns = np.ones((2,) + detector.shape)
        patterns[1] = 3

        sphere1 = _detector_to_stereo(detector, patterns[:1], rot[0], size=n)
        sphere2 = _detector_to_stereo(detector, patterns[1:], rot[1], size=n)
        sphere = _detector_to_stereo(detector, patterns, rot, size=n)

        both = np.isfinite(sphere1) & np.isfinite(sphere2)
        only1 = np.isfinite(sphere1) & np.isnan(sphere2)
        only2 = np.isnan(sphere1) & np.isfinite(sphere2)
        assert both.any() and only1.any() and only2.any()
        assert np.allclose(sphere[both], 2)
        assert np.allclose(sphere[only1], 1)
        assert np.allclose(sphere[only2], 3)
        assert np.isnan(sphere[~(both | only1 | only2)]).all()

    def test_dask_array(self, detector):
        n = 51
        rot = oqu.Rotation.from_euler(np.deg2rad([[10, 20, 30], [50, 40, 10]]))
        rot = oqu.Rotation(np.concatenate([rot.data, rot.data]))
        patterns = np.random.default_rng(42).random((4,) + detector.shape)
        sphere1 = _detector_to_stereo(detector, patterns, rot, size=n)
        sphere2 = _detector_to_stereo(
            detector, da.from_array(patterns, chunks=(3, -1, -1)), rot, size=n
        )
        assert np.allclose(sphere1, sphere2, equal_nan=True)

    @pytest.mark.parametrize("detector", [[(2,), (60, 60)]], indirect=["detector"])
    def test_pattern_specific_pcs(self, detector):
        n = 51
        detector.pcz = [0.5, 0.7]
        rot = oqu.Rotation.from_euler(np.deg2rad([[10, 20, 30], [10, 20, 30]]))
        patterns = np.random.default_rng(42).random((2,) + detector.shape)
        sphere = _detector_to_stereo(detector, patterns, rot, size=n)

        # Same as projecting each pattern with its own PC and combining
        accumulator = np.zeros((2, n, n))
        counts = np.zeros((2, n, n))
        for i in range(2):
            detector_i = kp.detectors.EBSDDetector(
                shape=detector.shape,
                binning=detector.binning,
                px_size=detector.px_size,
                pc=detector.pc[i],
                sample_tilt=detector.sample_tilt,
            )
            sphere_i = _detector_to_stereo(
                detector_i, patterns[i : i + 1], rot[i], size=n
            )
            is_valid = np.isfinite(sphere_i)
            accumulator[is_valid] += sphere_i[is_valid]
            counts[is_valid] += 1
        has_counts = counts > 0
        assert np.allclose(
            sphere[has_counts], accumulator[has_counts] / counts[has_counts]
        )
        assert np.isnan(sphere[~has_counts]).all()

    def test_symmetrized(self, detector):
        n = 51
        pg = kp.data.nickel_ebsd_master_pattern_small().phase.point_group
        ori = oqu.Orientation.from_euler(np.deg2rad([10, 20, 30]), symmetry=pg)
        patterns = np.ones((1,) + detector.shape)

        sphere1 = _detector_to_stereo(detector, patterns, ori, size=n)
        sphere2 = _detector_to_stereo(detector, patterns, ori, size=n, symmetrize=True)
        assert np.allclose(sphere2[np.isfinite(sphere2)], 1)

        # The 24 proper operators of m-3m cover (almost) the whole
        # sphere, while a single pattern covers only a small part
        _, _, inside_disk = _get_unit_sphere_vectors(n)
        n_disk = 2 * inside_disk.sum()
        assert np.isfinite(sphere1).sum() < 0.2 * n_disk
        assert np.isfinite(sphere2).sum() > 0.99 * n_disk

        # Unsymmetrized pixels are also covered when symmetrized
        assert np.isfinite(sphere2[np.isfinite(sphere1)]).all()

    def test_exclude_outside_equator(self, detector):
        n = 51
        _, _, inside_disk = _get_unit_sphere_vectors(n)
        outside = ~inside_disk.reshape(n, n)

        # Pattern crossing the equator
        rot = oqu.Rotation.from_axes_angles([1, 0, 0], -np.pi / 2)
        patterns = np.ones((1,) + detector.shape)

        sphere1 = _detector_to_stereo(detector, patterns, rot, size=n)
        sphere2 = _detector_to_stereo(
            detector, patterns, rot, size=n, exclude_outside_equator=False
        )
        assert np.isnan(sphere1[:, outside]).all()
        assert np.isfinite(sphere2[:, outside]).any()

        # Inside the disk the results are identical
        assert np.allclose(sphere1[:, ~outside], sphere2[:, ~outside], equal_nan=True)

    def test_progressbar(self, detector, capsys):
        patterns = np.ones((2,) + detector.shape)
        rot = oqu.Rotation.identity((2,))
        _ = _detector_to_stereo(detector, patterns, rot, size=11, show_progressbar=True)
        captured = capsys.readouterr()
        assert "2/2" in captured.err

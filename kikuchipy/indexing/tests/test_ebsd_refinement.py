# Copyright 2019-2023 The kikuchipy developers
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

import dask
import dask.array as da
from diffpy.structure import Atom, Lattice, Structure
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy.indexing._refinement._solvers import _prepare_pattern
from kikuchipy.signals.util._crystal_map import _equal_phase


class EBSDRefineTestSetup:
    nickel_ebsd_small = kp.data.nickel_ebsd_small()
    nickel_ebsd_small.remove_static_background()
    nickel_ebsd_small.remove_dynamic_background()

    mp = kp.signals.EBSDMasterPattern(
        np.random.rand(2, 5, 5, 5).astype(np.float32),
        axes=[
            dict(name="hemisphere", size=2, scale=1),
            dict(name="energy", size=5, offset=16, scale=1),
            dict(name="dy", size=5, scale=1),
            dict(name="dx", size=5, scale=1),
        ],
        projection="lambert",
        hemisphere="both",
        phase=Phase("a", 225),
    )


class TestEBSDRefine(EBSDRefineTestSetup):
    """Note that it is the calls to the :mod:`scipy.optimize` and
    NLopt methods that take up test time. The setup and array sizes do
    not matter that much.

    Tests relevant for all three refinement cases (orientation, PC and
    orientation/PC) goes in this class.
    """

    def test_prepare_pattern(self):
        pattern = np.linspace(0, 1, 100, dtype="float32")

        prepared_pattern1, squared_norm1 = _prepare_pattern(pattern, True)
        prepared_pattern2, squared_norm2 = _prepare_pattern.py_func(pattern, True)
        assert np.allclose(prepared_pattern1, prepared_pattern2)
        assert np.isclose(prepared_pattern1.mean(), 0, atol=1e-6)
        assert np.allclose(squared_norm1, squared_norm2)
        assert np.isclose(squared_norm1, 34.007, atol=1e-3)

        prepared_pattern3, squared_norm3 = _prepare_pattern(pattern, False)
        prepared_pattern4, squared_norm4 = _prepare_pattern.py_func(pattern, False)
        assert np.allclose(prepared_pattern3, prepared_pattern4)
        assert np.isclose(prepared_pattern3.mean(), 0, atol=1e-6)
        assert np.allclose(squared_norm3, squared_norm4)
        assert np.isclose(squared_norm3, 8.502, atol=1e-3)

    @pytest.mark.parametrize(
        "ebsd_with_axes_and_random_data, detector, error_msg",
        [
            (((2,), (3, 2), True, np.float32), ((2,), (2, 3)), r"Detector shape \(2, "),
            (((3,), (2, 3), True, np.float32), ((2,), (2, 3)), "Detector must have ex"),
        ],
        indirect=["ebsd_with_axes_and_random_data", "detector"],
    )
    def test_refine_check_raises(
        self,
        ebsd_with_axes_and_random_data,
        detector,
        error_msg,
        get_single_phase_xmap,
    ):
        s = ebsd_with_axes_and_random_data
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        with pytest.raises(ValueError, match=error_msg):
            _ = s.refine_orientation(
                xmap=xmap, master_pattern=self.mp, detector=detector, energy=20
            )

    def test_refine_raises(self, dummy_signal, get_single_phase_xmap):
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        detector = kp.detectors.EBSDDetector(shape=s._signal_shape_rc)
        refine_kwargs = dict(master_pattern=self.mp, energy=20, detector=detector)

        with pytest.raises(ValueError, match="Method 'a' not in the list of supported"):
            _ = s.refine_orientation(xmap=xmap, method="a", **refine_kwargs)

        with pytest.raises(ValueError, match=r"Signal mask shape \(10, 20\) and "):
            _ = s.refine_orientation(
                xmap=xmap, signal_mask=np.zeros((10, 20)), **refine_kwargs
            )

        xmap.phases.add(Phase(name="b", point_group="m-3m"))
        xmap._phase_id[0] = 1
        with pytest.raises(ValueError, match="Points in data in crystal map must have"):
            _ = s.refine_orientation(xmap=xmap, **refine_kwargs)

    def test_refine_signal_mask(self, dummy_signal, get_single_phase_xmap):
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        det = kp.detectors.EBSDDetector(shape=s._signal_shape_rc)
        ref_kw = dict(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=det,
            method="minimize",
            method_kwargs=dict(method="Nelder-Mead", options=dict(maxfev=10)),
        )
        xmap_ref_no_mask = s.refine_orientation(**ref_kw)
        signal_mask = np.zeros(s._signal_shape_rc, dtype=bool)
        signal_mask[0, 0] = 1  # Mask away upper left pixel

        xmap_ref_mask = s.refine_orientation(signal_mask=signal_mask, **ref_kw)

        assert not np.allclose(
            xmap_ref_no_mask.rotations.data, xmap_ref_mask.rotations.data
        )

    @pytest.mark.parametrize(
        "ebsd_with_axes_and_random_data, detector, rechunk, chunk_kwargs, chunksize",
        [
            (
                ((5, 4), (10, 8), True, np.float32),
                ((5, 4), (10, 8)),
                False,
                None,
                (20, 1),
            ),
            (
                ((5, 4), (10, 8), True, np.float32),
                ((5, 4), (10, 8)),
                True,
                dict(chunk_shape=3),
                (3, 1),
            ),
            (
                ((5, 4), (10, 8), True, np.float32),
                ((5, 4), (10, 8)),
                False,
                dict(chunk_shape=3),
                (20, 1),
            ),
        ],
        indirect=["ebsd_with_axes_and_random_data", "detector"],
    )
    def test_refine_orientation_chunking(
        self,
        ebsd_with_axes_and_random_data,
        detector,
        rechunk,
        chunk_kwargs,
        chunksize,
        get_single_phase_xmap,
    ):
        """Ensure the returned dask array when not computing has the
        desired chunksize.

        Ideally, the last dimension should have size 4 (score, phi1,
        Phi, phi2), but this requires better handling of removed and
        added axes and their sizes in the call to
        :func:`dask.array.map_blocks` in :func:`_refine_orientation` and
        the other equivalent private refinement functions.
        """
        s = ebsd_with_axes_and_random_data
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )

        dask_arr = s.refine_orientation(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=detector,
            compute=False,
            rechunk=rechunk,
            chunk_kwargs=chunk_kwargs,
        )
        assert dask_arr.chunksize == chunksize

    @pytest.mark.skipif(kp._nlopt_installed, reason="NLopt is installed")
    def test_refine_raises_nlopt_import_error(
        self, dummy_signal, get_single_phase_xmap
    ):  # pragma: no cover
        s = dummy_signal
        nav_shape = s._navigation_shape_rc
        xmap = get_single_phase_xmap(
            nav_shape=nav_shape,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        det = kp.detectors.EBSDDetector(shape=s._signal_shape_rc)

        with pytest.raises(ImportError, match="Package `nlopt`, required for method "):
            _ = s.refine_orientation_projection_center(
                xmap=xmap,
                master_pattern=self.mp,
                energy=20,
                detector=det,
                method="LN_NELDERMEAD",
            )

    @pytest.mark.skipif(not kp._nlopt_installed, reason="NLopt is not installed")
    def test_refine_raises_initial_step_nlopt(
        self, dummy_signal, get_single_phase_xmap
    ):
        s = dummy_signal
        nav_shape = s._navigation_shape_rc
        xmap = get_single_phase_xmap(
            nav_shape=nav_shape,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        det = kp.detectors.EBSDDetector(shape=s._signal_shape_rc)

        with pytest.raises(
            ValueError, match="The initial step must be a single number"
        ):
            _ = s.refine_orientation_projection_center(
                xmap=xmap,
                master_pattern=self.mp,
                energy=20,
                detector=det,
                method="LN_NELDERMEAD",
                initial_step=[1, 1, 1],
            )

    def test_refine_single_point(self, dummy_signal, get_single_phase_xmap):
        am = dummy_signal.axes_manager
        xmap = get_single_phase_xmap(
            nav_shape=am.navigation_shape[::-1],
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in am.navigation_axes)[::-1],
        )
        det = dummy_signal.detector.deepcopy()
        det.pc = det.pc_average

        nav_mask1 = np.ones(xmap.shape, dtype=bool)
        nav_mask1[0, 0] = False

        xmap_ref = dummy_signal.refine_orientation(
            xmap=xmap,
            detector=det,
            master_pattern=self.mp,
            energy=20,
            method_kwargs=dict(tol=0.1, options=dict(maxfev=10)),
            navigation_mask=nav_mask1,
        )
        assert xmap_ref.size == 1
        assert xmap_ref.shape == (1, 1)

        # Raises error when navigation mask is incorrect
        nav_mask2 = np.ones((xmap.shape[0] - 1, xmap.shape[1] - 1), dtype=bool)
        nav_mask2[0, 0] = False

        with pytest.raises(ValueError, match=r"Navigation mask shape \(2, 2\) and "):
            _ = dummy_signal.refine_orientation(
                xmap=xmap,
                detector=det,
                master_pattern=self.mp,
                energy=20,
                navigation_mask=nav_mask2,
            )

    def test_equal_phase(self):
        assert _equal_phase(Phase(), Phase()) == (True, None)

        # Name
        assert _equal_phase(Phase("a"), Phase("a")) == (True, None)
        assert _equal_phase(Phase("a"), Phase("b")) == (False, "names")

        # Space group
        assert _equal_phase(Phase(space_group=1), Phase(space_group=1)) == (True, None)
        assert _equal_phase(Phase(), Phase(space_group=2)) == (False, "space groups")
        assert _equal_phase(Phase(space_group=1), Phase(space_group=2)) == (
            False,
            "space groups",
        )

        # Point group
        assert _equal_phase(Phase(), Phase(point_group="m-3m")) == (
            False,
            "point groups",
        )
        assert _equal_phase(Phase(point_group="4"), Phase(point_group="m-3m")) == (
            False,
            "point groups",
        )

        # Structure
        atom_al = Atom("Al", [0, 0, 0])
        atom_al2 = Atom("Al", [0.5, 0.5, 0.5])
        atom_al3 = Atom("Al", [0, 0, 0], occupancy=0.5)
        atom_mn = Atom("Mn", [0, 0, 0])
        assert _equal_phase(
            Phase(structure=Structure(atoms=[atom_al, atom_mn])),
            Phase(structure=Structure(atoms=[atom_al, atom_mn])),
        ) == (True, None)
        assert _equal_phase(
            Phase(structure=Structure(atoms=[atom_al])),
            Phase(structure=Structure(atoms=[atom_al, atom_mn])),
        ) == (False, "number of atoms")
        assert _equal_phase(
            Phase(structure=Structure(atoms=[atom_al2, atom_mn])),
            Phase(structure=Structure(atoms=[atom_al, atom_mn])),
        ) == (False, "atoms")
        assert _equal_phase(
            Phase(structure=Structure(atoms=[atom_al, atom_mn])),
            Phase(structure=Structure(atoms=[atom_al3, atom_mn])),
        ) == (False, "atoms")

        # Lattice
        assert _equal_phase(
            Phase(structure=Structure(lattice=Lattice(1, 2, 3, 90, 100, 110))),
            Phase(structure=Structure(lattice=Lattice(1, 2, 3, 90, 100, 110))),
        ) == (True, None)
        assert _equal_phase(
            Phase(structure=Structure(lattice=Lattice(1, 2, 4, 90, 100, 110))),
            Phase(structure=Structure(lattice=Lattice(1, 2, 3, 90, 100, 110))),
        ) == (False, "lattice parameters")

    def test_refinement_invalid_phase(self, dummy_signal, get_single_phase_xmap):
        am = dummy_signal.axes_manager
        xmap = get_single_phase_xmap(
            nav_shape=am.navigation_shape[::-1],
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in am.navigation_axes)[::-1],
        )
        xmap.phases[0].name = "b"
        det = dummy_signal.detector.deepcopy()
        det.pc = det.pc_average

        with pytest.raises(ValueError, match="Master pattern phase 'a' and phase of "):
            _ = dummy_signal.refine_orientation(
                xmap=xmap, detector=det, master_pattern=self.mp, energy=20
            )


class TestEBSDRefineOrientation(EBSDRefineTestSetup):
    @pytest.mark.parametrize(
        "ebsd_with_axes_and_random_data, detector, method_kwargs, trust_region",
        [
            (
                ((2,), (2, 3), True, np.float32),
                ((2,), (2, 3)),
                dict(method="Nelder-Mead"),
                None,
            ),
            (
                ((3, 2), (2, 3), False, np.uint8),
                ((1,), (2, 3)),
                dict(method="Powell"),
                [1, 1, 1],
            ),
        ],
        indirect=["ebsd_with_axes_and_random_data", "detector"],
    )
    def test_refine_orientation_local(
        self,
        ebsd_with_axes_and_random_data,
        detector,
        method_kwargs,
        trust_region,
        get_single_phase_xmap,
    ):
        s = ebsd_with_axes_and_random_data
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        method_kwargs.update(dict(options=dict(maxfev=10)))

        xmap_ref = s.refine_orientation(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=detector,
            trust_region=trust_region,
            method_kwargs=method_kwargs,
        )
        assert xmap_ref.shape == xmap.shape
        assert not np.allclose(xmap_ref.rotations.data, xmap.rotations.data)

    @pytest.mark.parametrize(
        (
            "ebsd_with_axes_and_random_data, detector, method, initial_step, rtol, "
            "maxeval, trust_region"
        ),
        [
            (
                ((2,), (2, 3), True, np.float32),
                ((2,), (2, 3)),
                "LN_NELDERMEAD",
                None,
                1e-3,
                20,
                None,
            ),
            (
                ((3, 2), (2, 3), False, np.uint8),
                ((1,), (2, 3)),
                "LN_NELDERMEAD",
                1,
                1e-2,
                10,
                [1, 1, 1],
            ),
        ],
        indirect=["ebsd_with_axes_and_random_data", "detector"],
    )
    @pytest.mark.skipif(not kp._nlopt_installed, reason="NLopt is not installed")
    def test_refine_orientation_local_nlopt(
        self,
        ebsd_with_axes_and_random_data,
        detector,
        method,
        initial_step,
        rtol,
        maxeval,
        trust_region,
        get_single_phase_xmap,
    ):
        s = ebsd_with_axes_and_random_data
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )

        xmap_ref = s.refine_orientation(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=detector,
            method=method,
            trust_region=trust_region,
            initial_step=initial_step,
            rtol=rtol,
            maxeval=maxeval,
        )
        assert xmap_ref.shape == xmap.shape
        assert not np.allclose(xmap_ref.rotations.data, xmap.rotations.data)

    def test_refine_orientation_not_compute(
        self,
        dummy_signal,
        get_single_phase_xmap,
    ):
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        det = kp.detectors.EBSDDetector(shape=s._signal_shape_rc)

        dask_arr = s.refine_orientation(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=det,
            method_kwargs=dict(options=dict(maxfev=10)),
            compute=False,
        )
        assert isinstance(dask_arr, da.Array)
        assert dask.is_dask_collection(dask_arr)
        # Should ideally be (9, 4) with better use of map_blocks()
        assert dask_arr.shape == (9, 1)

    @pytest.mark.parametrize(
        "method, method_kwargs",
        [
            (
                "basinhopping",
                dict(minimizer_kwargs=dict(method="Nelder-Mead"), niter=1),
            ),
            ("differential_evolution", dict(maxiter=1)),
            ("dual_annealing", dict(maxiter=1)),
            (
                "shgo",
                dict(
                    sampling_method="sobol",
                    options=dict(f_tol=1e-3, maxfev=1),
                    minimizer_kwargs=dict(
                        method="Nelder-Mead", options=dict(fatol=1e-3)
                    ),
                ),
            ),
        ],
    )
    def test_refine_orientation_global(
        self,
        method,
        method_kwargs,
        ebsd_with_axes_and_random_data,
        get_single_phase_xmap,
    ):
        s = ebsd_with_axes_and_random_data
        det = kp.detectors.EBSDDetector(shape=s._signal_shape_rc)
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        xmap_ref = s.refine_orientation(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=det,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=(0.5, 0.5, 0.5),
        )
        assert xmap_ref.shape == xmap.shape
        assert not np.allclose(xmap_ref.rotations.data, xmap.rotations.data)

    def test_refine_orientation_nickel_ebsd_small(self):
        """Refine already refined orientations with SciPy, which should
        produce comparable results.
        """
        s = self.nickel_ebsd_small

        energy = 20
        signal_mask = kp.filters.Window("circular", s._signal_shape_rc)
        signal_mask = ~signal_mask.astype(bool)
        xmap_ref = s.refine_orientation(
            xmap=s.xmap,
            detector=s.detector,
            master_pattern=kp.data.nickel_ebsd_master_pattern_small(
                energy=energy, projection="lambert"
            ),
            energy=energy,
            signal_mask=signal_mask,
        )
        assert np.allclose(xmap_ref.scores, s.xmap.scores, atol=1e-3)

    @pytest.mark.skipif(not kp._nlopt_installed, reason="NLopt is not installed")
    def test_refine_orientation_nickel_ebsd_small_nlopt(self):
        """Refine already refined orientations with NLopt, which should
        produce slightly better results.
        """
        s = self.nickel_ebsd_small

        energy = 20
        signal_mask = kp.filters.Window("circular", s._signal_shape_rc)
        signal_mask = ~signal_mask.astype(bool)

        xmap_ref = s.refine_orientation(
            xmap=s.xmap,
            detector=s.detector,
            master_pattern=kp.data.nickel_ebsd_master_pattern_small(
                energy=energy,
                projection="lambert",
            ),
            energy=energy,
            signal_mask=signal_mask,
            method="LN_NELDERMEAD",
            trust_region=[2, 2, 2],
        )
        assert xmap_ref.scores.mean() > s.xmap.scores.mean()

    @pytest.mark.skipif(not kp._nlopt_installed, reason="NLopt is not installed")
    def test_refine_orientation_pseudo_symmetry_nlopt(self):
        s = self.nickel_ebsd_small

        energy = 20
        signal_mask = kp.filters.Window("circular", s._signal_shape_rc)
        signal_mask = ~signal_mask.astype(bool)

        rot_ps = Rotation.from_axes_angles([[0, 0, 1], [0, 0, -1]], np.deg2rad(30))

        # Apply the first rotation so that the second rotation (the
        # inverse) is the best match
        xmap = s.xmap.deepcopy()
        xmap._rotations[0] = (rot_ps[0] * xmap.rotations[0]).data

        xmap_ref = s.refine_orientation(
            xmap=xmap,
            detector=s.detector,
            master_pattern=kp.data.nickel_ebsd_master_pattern_small(
                energy=energy,
                projection="lambert",
            ),
            energy=energy,
            signal_mask=signal_mask,
            method="LN_NELDERMEAD",
            trust_region=[2, 2, 2],
            pseudo_symmetry_ops=rot_ps,
        )
        assert xmap_ref.scores.mean() > xmap.scores.mean()
        assert np.allclose(xmap_ref.pseudo_symmetry_index, [2, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_refine_orientation_pseudo_symmetry_scipy(self):
        s = self.nickel_ebsd_small

        energy = 20
        signal_mask = kp.filters.Window("circular", s._signal_shape_rc)
        signal_mask = ~signal_mask.astype(bool)

        rot_ps = Rotation.from_axes_angles([[0, 0, 1], [0, 0, -1]], np.deg2rad(30))

        # Apply the second rotation so that the first rotation (the
        # inverse) is the best match
        xmap = s.xmap.deepcopy()
        xmap._rotations[5] = (rot_ps[1] * xmap.rotations[5]).data

        ref_kw = dict(
            xmap=xmap,
            detector=s.detector,
            master_pattern=kp.data.nickel_ebsd_master_pattern_small(
                energy=energy,
                projection="lambert",
            ),
            energy=energy,
            signal_mask=signal_mask,
            pseudo_symmetry_ops=rot_ps,
            trust_region=[2, 2, 2],
        )

        # Nelder-Mead
        xmap_ref = s.refine_orientation(**ref_kw)
        assert xmap_ref.scores.mean() > xmap.scores.mean()
        assert np.allclose(xmap_ref.pseudo_symmetry_index, [0, 0, 0, 0, 0, 1, 0, 0, 0])

        # Global: Basin-hopping
        nav_mask = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=bool).reshape(xmap.shape)
        _ = s.refine_orientation(
            method="basinhopping",
            method_kwargs=dict(minimizer_kwargs=dict(method="Nelder-Mead"), niter=1),
            navigation_mask=nav_mask,
            **ref_kw,
        )

        # Global: Differential evolution
        _ = s.refine_orientation(
            method="differential_evolution",
            navigation_mask=nav_mask,
            **ref_kw,
        )

    def test_refine_orientation_not_indexed_case1(
        self, dummy_signal, get_single_phase_xmap
    ):
        """Test refinining crystal map with some points considered
        not-indexed (phase ID of -1).
        """
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        xmap.phases.add_not_indexed()
        xmap[1, 2].phase_id = -1

        ref_kw = dict(detector=s.detector, master_pattern=self.mp, energy=20)

        xmap_ref = s.refine_orientation(xmap, **ref_kw)

        assert xmap_ref.size == 9
        assert np.allclose(xmap_ref.is_in_data, xmap.is_in_data)
        assert np.allclose(xmap_ref.phase_id, xmap.phase_id)
        assert "not_indexed" in xmap_ref.phases.names

    def test_refine_orientation_not_indexed_case2(
        self, dummy_signal, get_single_phase_xmap
    ):
        """Test refinining crystal map with some points considered
        not-indexed (phase ID of -1) within a navigation mask.
        """
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        xmap.phases.add_not_indexed()
        xmap[1, 2].phase_id = -1

        nav_mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        xmap_ref = s.refine_orientation(
            xmap, s.detector, self.mp, 20, navigation_mask=nav_mask.reshape(xmap.shape)
        )

        assert xmap_ref.size == 7
        assert np.allclose(xmap_ref.is_in_data, ~nav_mask)
        assert np.allclose(xmap_ref.phase_id, xmap.phase_id[~nav_mask])
        assert "not_indexed" in xmap_ref.phases.names

    def test_refine_orientation_not_indexed_case3(
        self, dummy_signal, get_single_phase_xmap
    ):
        """Test refinining crystal map with some points considered
        not-indexed (phase ID of -1) outside a navigation mask.
        """
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        xmap.phases.add_not_indexed()
        xmap[1, 2].phase_id = -1

        nav_mask = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
        xmap_ref = s.refine_orientation(
            xmap,
            s.detector,
            self.mp,
            20,
            navigation_mask=nav_mask.reshape((3, 3)),
        )

        assert xmap_ref.size == 8
        assert np.allclose(xmap_ref.is_in_data, ~nav_mask)
        assert np.allclose(xmap_ref.phase_id, xmap.phase_id[~nav_mask])
        assert "not_indexed" not in xmap_ref.phases_in_data.names

    def test_refine_orientation_not_indexed_case4(
        self, dummy_signal, get_single_phase_xmap
    ):
        """Test refinining crystal map with some points considered
        not-indexed (phase ID of -1) and not in the data.
        """
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        xmap.phases.add_not_indexed()
        xmap.is_in_data[[5, 7]] = False
        xmap.phases.add_not_indexed()
        xmap[0, 0].phase_id = -1

        xmap_ref = s.refine_orientation(xmap, s.detector, self.mp, 20)

        assert xmap_ref.size == 7
        assert np.allclose(xmap_ref.is_in_data, xmap.is_in_data)
        assert np.allclose(xmap_ref.phase_id, xmap.phase_id)
        assert "not_indexed" in xmap_ref.phases_in_data.names


class TestEBSDRefinePC(EBSDRefineTestSetup):
    @pytest.mark.parametrize(
        "ebsd_with_axes_and_random_data, detector, method_kwargs, trust_region",
        [
            (
                ((4,), (3, 4), True, np.float32),
                ((4,), (3, 4)),
                dict(method="Nelder-Mead"),
                None,
            ),
            (
                ((3, 2), (2, 3), False, np.uint8),
                ((1,), (2, 3)),
                dict(method="Powell"),
                [0.01, 0.01, 0.01],
            ),
        ],
        indirect=["ebsd_with_axes_and_random_data", "detector"],
    )
    def test_refine_projection_center_local(
        self,
        ebsd_with_axes_and_random_data,
        detector,
        method_kwargs,
        trust_region,
        get_single_phase_xmap,
    ):
        s = ebsd_with_axes_and_random_data
        nav_shape = s._navigation_shape_rc
        xmap = get_single_phase_xmap(
            nav_shape=nav_shape,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        method_kwargs.update(dict(options=dict(maxfev=10)))
        signal_mask = np.zeros(detector.shape, dtype=bool)

        scores_ref, det_ref, num_evals_ref = s.refine_projection_center(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=detector,
            signal_mask=signal_mask,
            trust_region=trust_region,
            method_kwargs=method_kwargs,
        )
        assert scores_ref.shape == nav_shape
        assert not np.allclose(xmap.get_map_data("scores"), scores_ref)
        assert isinstance(det_ref, kp.detectors.EBSDDetector)
        assert det_ref.pc.shape == nav_shape + (3,)
        assert num_evals_ref.shape == nav_shape

        # TODO: Change to == 10 once Python 3.7 is unsopprted.
        assert num_evals_ref.max() < 50

    @pytest.mark.parametrize(
        (
            "ebsd_with_axes_and_random_data, detector, method, initial_step, rtol, "
            "maxeval, trust_region"
        ),
        [
            (
                ((4,), (3, 4), True, np.float32),
                ((4,), (3, 4)),
                "LN_NELDERMEAD",
                None,
                1e-3,
                20,
                None,
            ),
            (
                ((3, 2), (2, 3), False, np.uint8),
                ((1,), (2, 3)),
                "LN_NELDERMEAD",
                0.05,
                1e-2,
                None,
                [0.02, 0.02, 0.02],
            ),
        ],
        indirect=["ebsd_with_axes_and_random_data", "detector"],
    )
    @pytest.mark.skipif(not kp._nlopt_installed, reason="NLopt is not installed")
    def test_refine_projection_center_local_nlopt(
        self,
        ebsd_with_axes_and_random_data,
        detector,
        method,
        initial_step,
        rtol,
        maxeval,
        trust_region,
        get_single_phase_xmap,
    ):
        s = ebsd_with_axes_and_random_data
        nav_shape = s._navigation_shape_rc
        xmap = get_single_phase_xmap(
            nav_shape=nav_shape,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        signal_mask = np.zeros(detector.shape, dtype=bool)

        scores_ref, det_ref, num_evals_ref = s.refine_projection_center(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=detector,
            signal_mask=signal_mask,
            method=method,
            trust_region=trust_region,
            rtol=rtol,
            maxeval=maxeval,
            initial_step=initial_step,
        )
        assert scores_ref.shape == nav_shape
        assert not np.allclose(xmap.get_map_data("scores"), scores_ref)
        assert isinstance(det_ref, kp.detectors.EBSDDetector)
        assert det_ref.pc.shape == nav_shape + (3,)
        assert num_evals_ref.shape == nav_shape
        if maxeval:
            assert num_evals_ref.max() == maxeval

    @pytest.mark.parametrize(
        "method, method_kwargs",
        [
            (
                "basinhopping",
                dict(minimizer_kwargs=dict(method="Nelder-Mead"), niter=1),
            ),
            ("basinhopping", None),
            ("differential_evolution", dict(maxiter=1)),
            ("dual_annealing", dict(maxiter=1)),
            (
                "shgo",
                dict(
                    sampling_method="sobol",
                    options=dict(f_tol=1e-3, maxfev=1),
                    minimizer_kwargs=dict(
                        method="Nelder-Mead", options=dict(fatol=1e-3)
                    ),
                ),
            ),
        ],
    )
    def test_refine_projection_center_global(
        self,
        method,
        method_kwargs,
        ebsd_with_axes_and_random_data,
        get_single_phase_xmap,
    ):
        s = ebsd_with_axes_and_random_data
        detector = kp.detectors.EBSDDetector(shape=s._signal_shape_rc)
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )

        scores_ref, det_ref, num_evals_ref = s.refine_projection_center(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=detector,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=(0.01, 0.01, 0.01),
        )
        assert scores_ref.shape == xmap.shape
        assert not np.allclose(scores_ref, xmap.get_map_data("scores"))
        assert isinstance(det_ref, kp.detectors.EBSDDetector)
        assert num_evals_ref.shape == xmap.shape

    def test_refine_projection_center_not_compute(
        self,
        dummy_signal,
        get_single_phase_xmap,
    ):
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        det = kp.detectors.EBSDDetector(shape=s._signal_shape_rc)

        dask_arr = s.refine_projection_center(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=det,
            method_kwargs=dict(options=dict(maxfev=10)),
            compute=False,
        )
        assert isinstance(dask_arr, da.Array)
        assert dask.is_dask_collection(dask_arr)
        # Should ideally be (9, 5) with better use of map_blocks()
        assert dask_arr.shape == (9, 1)


class TestEBSDRefineOrientationPC(EBSDRefineTestSetup):
    @pytest.mark.parametrize(
        "method_kwargs, trust_region",
        [
            (dict(method="Nelder-Mead"), None),
            (dict(method="Powell"), [0.5, 0.5, 0.5, 0.01, 0.01, 0.01]),
        ],
    )
    def test_refine_orientation_projection_center_local(
        self,
        dummy_signal,
        method_kwargs,
        trust_region,
        get_single_phase_xmap,
    ):
        s = dummy_signal
        nav_shape = s._navigation_shape_rc
        xmap = get_single_phase_xmap(
            nav_shape=nav_shape,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        det = kp.detectors.EBSDDetector(shape=s._signal_shape_rc)
        method_kwargs.update(dict(options=dict(maxfev=10)))
        signal_mask = np.zeros(det.shape, dtype=bool)

        xmap_ref, det_ref = s.refine_orientation_projection_center(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=det,
            signal_mask=signal_mask,
            trust_region=trust_region,
            method_kwargs=method_kwargs,
        )
        assert xmap_ref.shape == xmap.shape
        assert not np.allclose(xmap_ref.rotations.data, xmap.rotations.data)
        assert isinstance(det_ref, kp.detectors.EBSDDetector)
        assert det_ref.pc.shape == nav_shape + (3,)

    @pytest.mark.parametrize(
        "method, trust_region, rtol, initial_step, maxeval",
        [
            ("LN_NELDERMEAD", None, 1e-3, None, 50),
            ("LN_NELDERMEAD", [0.5, 0.5, 0.5, 0.01, 0.01, 0.01], 1e-4, [1, 0.02], None),
        ],
    )
    @pytest.mark.skipif(not kp._nlopt_installed, reason="NLopt is not installed")
    def test_refine_orientation_projection_center_local_nlopt(
        self,
        dummy_signal,
        method,
        trust_region,
        rtol,
        initial_step,
        maxeval,
        get_single_phase_xmap,
    ):  # pragma: no cover
        s = dummy_signal
        nav_shape = s._navigation_shape_rc
        xmap = get_single_phase_xmap(
            nav_shape=nav_shape,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        det = kp.detectors.EBSDDetector(shape=s._signal_shape_rc)
        signal_mask = np.zeros(det.shape, dtype=bool)

        xmap_ref, det_ref = s.refine_orientation_projection_center(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=det,
            signal_mask=signal_mask,
            method=method,
            trust_region=trust_region,
            rtol=rtol,
            initial_step=initial_step,
            maxeval=maxeval,
        )
        assert xmap_ref.shape == xmap.shape
        assert not np.allclose(xmap_ref.rotations.data, xmap.rotations.data)
        assert isinstance(det_ref, kp.detectors.EBSDDetector)
        assert det_ref.pc.shape == nav_shape + (3,)

    @pytest.mark.parametrize(
        "method, method_kwargs",
        [
            (
                "basinhopping",
                dict(minimizer_kwargs=dict(method="Nelder-Mead"), niter=1),
            ),
            ("differential_evolution", dict(maxiter=1)),
            ("dual_annealing", dict(maxiter=1)),
            (
                "shgo",
                dict(
                    sampling_method="sobol",
                    options=dict(f_tol=1e-3, maxfev=1),
                    minimizer_kwargs=dict(
                        method="Nelder-Mead", options=dict(fatol=1e-3)
                    ),
                ),
            ),
        ],
    )
    def test_refine_orientation_projection_center_global(
        self,
        method,
        method_kwargs,
        dummy_signal,
        get_single_phase_xmap,
    ):
        s = dummy_signal
        det = kp.detectors.EBSDDetector(shape=s._signal_shape_rc)
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )

        xmap_ref, det_ref = s.refine_orientation_projection_center(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=det,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=[0.5, 0.5, 0.5, 0.01, 0.01, 0.01],
        )
        assert xmap_ref.shape == xmap.shape
        assert not np.allclose(xmap_ref.rotations.data, xmap.rotations.data)
        assert isinstance(det_ref, kp.detectors.EBSDDetector)
        assert not np.allclose(det.pc, det_ref.pc[0, 0])

    def test_refine_orientation_projection_center_not_compute(
        self, dummy_signal, get_single_phase_xmap
    ):
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        det = kp.detectors.EBSDDetector(shape=s._signal_shape_rc)

        dask_array = s.refine_orientation_projection_center(
            xmap=xmap,
            master_pattern=self.mp,
            energy=20,
            detector=det,
            method_kwargs=dict(options=dict(maxfev=1)),
            compute=False,
        )
        assert isinstance(dask_array, da.Array)
        assert dask.is_dask_collection(dask_array)
        # Should ideally be (9, 8) with better use of map_blocks()
        assert dask_array.shape == (9, 1)

    @pytest.mark.skipif(not kp._nlopt_installed, reason="NLopt is not installed")
    def test_refine_orientation_pc_pseudo_symmetry_nlopt(self):
        s = self.nickel_ebsd_small

        energy = 20
        signal_mask = kp.filters.Window("circular", s._signal_shape_rc)
        signal_mask = ~signal_mask.astype(bool)

        rot_ps = Rotation.from_axes_angles([[0, 0, 1], [0, 0, -1]], np.deg2rad(30))

        # Apply the first rotation so that the second rotation (the
        # inverse) is the best match
        xmap = s.xmap.deepcopy()
        xmap._rotations[0] = (rot_ps[0] * xmap.rotations[0]).data

        xmap_ref, det_ref = s.refine_orientation_projection_center(
            xmap=xmap,
            detector=s.detector,
            master_pattern=kp.data.nickel_ebsd_master_pattern_small(
                energy=energy,
                projection="lambert",
            ),
            energy=energy,
            signal_mask=signal_mask,
            method="LN_NELDERMEAD",
            trust_region=[2, 2, 2, 0.05, 0.05, 0.05],
            pseudo_symmetry_ops=rot_ps,
        )
        assert xmap_ref.scores.mean() > xmap.scores.mean()
        assert np.allclose(xmap_ref.pseudo_symmetry_index, [2, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_refine_orientation_pc_pseudo_symmetry_scipy(self):
        s = self.nickel_ebsd_small

        energy = 20
        signal_mask = kp.filters.Window("circular", s._signal_shape_rc)
        signal_mask = ~signal_mask.astype(bool)

        rot_ps = Rotation.from_axes_angles([[0, 0, 1], [0, 0, -1]], np.deg2rad(30))

        # Apply the second rotation so that the first rotation (the
        # inverse) is the best match
        xmap = s.xmap.deepcopy()
        xmap._rotations[5] = (rot_ps[1] * xmap.rotations[5]).data

        ref_kw = dict(
            xmap=xmap,
            detector=s.detector,
            master_pattern=kp.data.nickel_ebsd_master_pattern_small(
                energy=energy,
                projection="lambert",
            ),
            energy=energy,
            signal_mask=signal_mask,
            pseudo_symmetry_ops=rot_ps,
        )

        # Nelder-Mead
        xmap_ref, det_ref = s.refine_orientation_projection_center(
            trust_region=[2, 2, 2, 0.05, 0.05, 0.05], **ref_kw
        )
        assert xmap_ref.scores.mean() > xmap.scores.mean()
        assert np.allclose(xmap_ref.pseudo_symmetry_index, [0, 0, 0, 0, 0, 1, 0, 0, 0])

        # Global: Basin-hopping
        nav_mask = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=bool).reshape(xmap.shape)
        _, _ = s.refine_orientation_projection_center(
            method="basinhopping",
            method_kwargs=dict(minimizer_kwargs=dict(method="Nelder-Mead"), niter=1),
            navigation_mask=nav_mask,
            **ref_kw,
        )

        # Global: Differential evolution
        _, _ = s.refine_orientation_projection_center(
            method="differential_evolution",
            navigation_mask=nav_mask,
            **ref_kw,
        )

    def test_refine_orientation_pc_not_indexed_case1(
        self, dummy_signal, get_single_phase_xmap
    ):
        """Test refinining orientations and PC with one point
        considered not-indexed (phase ID of -1).
        """
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        xmap.phases.add_not_indexed()
        xmap[1, 2].phase_id = -1

        xmap_ref, det_ref = s.refine_orientation_projection_center(
            xmap, s.detector, self.mp, 20
        )

        assert xmap_ref.size == 9
        assert np.all(xmap_ref.is_in_data)
        assert np.allclose(xmap_ref.phase_id, xmap.phase_id)
        assert "not_indexed" in xmap_ref.phases.names
        assert det_ref.navigation_shape == (8,)

    def test_refine_orientation_pc_not_indexed_case2(
        self, dummy_signal, get_single_phase_xmap
    ):
        """Test refinining orientations and PC with one point
        considered not-indexed (phase ID of -1) within a navigation
        mask.
        """
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        xmap.phases.add_not_indexed()
        xmap[1, 2].phase_id = -1

        nav_mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        xmap_ref, det_ref = s.refine_orientation_projection_center(
            xmap, s.detector, self.mp, 20, navigation_mask=nav_mask.reshape(xmap.shape)
        )

        assert xmap_ref.size == 7
        assert np.allclose(xmap_ref.is_in_data, ~nav_mask)
        assert np.allclose(xmap_ref.phase_id, xmap.phase_id[~nav_mask])
        assert "not_indexed" in xmap_ref.phases.names
        assert det_ref.navigation_shape == (6,)

    def test_refine_orientation_pc_not_indexed_case3(
        self, dummy_signal, get_single_phase_xmap
    ):
        """Test refinining orientations and PC with one point
        considered not-indexed (phase ID of -1) outside a navigation
        mask.
        """
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        xmap.phases.add_not_indexed()
        xmap[1, 2].phase_id = -1

        nav_mask = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
        xmap_ref, det_ref = s.refine_orientation_projection_center(
            xmap, s.detector, self.mp, 20, navigation_mask=nav_mask.reshape((3, 3))
        )

        assert xmap_ref.size == 8
        assert np.allclose(xmap_ref.is_in_data, ~nav_mask)
        assert np.allclose(xmap_ref.phase_id, xmap.phase_id[~nav_mask])
        assert "not_indexed" not in xmap_ref.phases_in_data.names
        assert det_ref.navigation_shape == (8,)

    def test_refine_orientation_pc_not_indexed_case4(
        self, dummy_signal, get_single_phase_xmap
    ):
        """Test refinining orientations and PCs with one point
        considered not-indexed (phase ID of -1) and some points not in
        the data.
        """
        s = dummy_signal
        xmap = get_single_phase_xmap(
            nav_shape=s._navigation_shape_rc,
            rotations_per_point=1,
            step_sizes=tuple(a.scale for a in s.axes_manager.navigation_axes)[::-1],
        )
        xmap.phases.add_not_indexed()
        xmap.is_in_data[[5, 7]] = False
        xmap.phases.add_not_indexed()
        xmap[0, 0].phase_id = -1

        xmap_ref, det_ref = s.refine_orientation_projection_center(
            xmap, s.detector, self.mp, 20
        )

        assert xmap_ref.size == 7
        assert np.allclose(xmap_ref.is_in_data, xmap.is_in_data)
        assert np.allclose(xmap_ref.phase_id, xmap.phase_id)
        assert "not_indexed" in xmap_ref.phases_in_data.names
        assert det_ref.navigation_shape == (6,)

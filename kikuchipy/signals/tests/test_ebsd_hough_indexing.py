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

from diffpy.structure import Lattice, Structure
from diffsims.crystallography import ReciprocalLatticeVector
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList
import pytest

import kikuchipy as kp
from kikuchipy.indexing._hough_indexing import (
    _get_info_message,
    _indexer_is_compatible_with_kikuchipy,
    _phase_lists_are_compatible,
)


@pytest.mark.skipif(
    not kp._pyebsdindex_installed, reason="pyebsdindex is not installed"
)
class TestHoughIndexing:
    def setup_method(self):
        s = kp.data.nickel_ebsd_small()
        s.remove_static_background()
        s.remove_dynamic_background()

        self.signal = s
        self.indexer = s.detector.get_indexer(s.xmap.phases)

    def test_hough_indexing_print_information(self):
        det = self.signal.detector
        det.shape = (4, 5)  # Save time on indexer creation

        indexer1 = det.get_indexer(self.signal.xmap.phases)

        info = _get_info_message(
            self.signal.axes_manager.navigation_size, chunksize=3, indexer=indexer1
        )
        info_list = info.split("\n")
        # fmt: off
        assert info_list[0] ==     "Hough indexing with PyEBSDIndex information:"
        assert info_list[1][:12] == "  PyOpenCL: "
        assert info_list[2] ==     "  Projection center (Bruker, mean): (0.4251, 0.2134, 0.5007)"
        assert info_list[3] ==     "  Indexing 9 pattern(s) in 3 chunk(s)"
        # fmt: on

        det_pc_mean = det.deepcopy()
        det_pc_mean.pc = det_pc_mean.pc_average
        indexer2 = det_pc_mean.get_indexer(self.signal.xmap.phases)
        info2 = _get_info_message(
            self.signal.axes_manager.navigation_size, chunksize=3, indexer=indexer2
        )
        info_list2 = info2.split("\n")
        assert info_list2[2] == "  Projection center (Bruker): (0.4251, 0.2134, 0.5007)"

    def test_hough_indexing(self):
        # Reference results (Hough indexing + refinement)
        xmap_ref = self.signal.xmap

        phase_list = xmap_ref.phases
        indexer = self.indexer

        xmap = self.signal.hough_indexing(phase_list, indexer)

        assert isinstance(xmap, CrystalMap)
        assert xmap.phases.names == phase_list.names
        angles = xmap.orientations.angle_with(xmap_ref.orientations, degrees=True)
        assert np.all(angles < 1)

    def test_hough_indexing_lazy(self):  # pragma: no cover
        s = self.signal.as_lazy()

        phase_list = self.signal.xmap.phases
        if kp._pyopencl_context_available:
            xmap1 = s.hough_indexing(phase_list, self.indexer)
            xmap2 = self.signal.hough_indexing(phase_list, self.indexer)
            assert np.allclose(xmap1.rotations.data, xmap2.rotations.data)
            assert np.allclose(xmap1.fit, xmap2.fit)
        else:
            with pytest.raises(ValueError, match="Hough indexing of lazy signals must"):
                _ = s.hough_indexing(phase_list, self.indexer, verbose=2)

    def test_hough_indexing_return_index_data(self):
        phase_list = self.signal.xmap.phases
        xmap, index_data = self.signal.hough_indexing(
            phase_list, self.indexer, return_index_data=True
        )

        index_data_dtypes = index_data.dtype.fields.keys()
        for field in ["quat", "iq", "pq", "cm", "phase", "fit", "nmatch"]:
            assert field in index_data_dtypes
        assert index_data.shape == (
            len(self.indexer.phaselist) + 1,
            self.signal.axes_manager.navigation_size,
        )
        xmap2 = kp.indexing.xmap_from_hough_indexing_data(index_data, phase_list)
        assert xmap2.shape == (9,)
        assert np.allclose(xmap2.rotations.data, xmap.rotations.data)
        assert np.allclose(xmap2.fit, xmap.fit)
        assert np.allclose(xmap2.cm, xmap.cm)

    def test_hough_indexing_return_band_data(self):
        phase_list = self.signal.xmap.phases
        indexer = self.signal.detector.get_indexer(phase_list, nBands=8)
        _, band_data = self.signal.hough_indexing(
            phase_list, indexer, return_band_data=True
        )
        assert isinstance(band_data, np.ndarray)
        assert band_data.shape == (
            self.signal.axes_manager.navigation_size,
            indexer.bandDetectPlan.nBands,
        )

        _, index_data, band_data = self.signal.hough_indexing(
            phase_list, self.indexer, return_index_data=True, return_band_data=True
        )
        assert isinstance(index_data, np.ndarray)
        assert isinstance(band_data, np.ndarray)
        assert index_data.shape == (2, 9)
        assert band_data.shape == (9, 9)

    def test_hough_indexing_raises_dissimilar_phase_lists(self):
        phase_list = PhaseList(names=["a", "b"], space_groups=[225, 229])
        with pytest.raises(
            ValueError, match=r"`phase_list` \(2\) and `indexer.phaselist` \(1\) have "
        ):
            _ = self.signal.hough_indexing(phase_list, self.indexer)

    def test_indexer_is_compatible_with_signal(self):
        indexer = self.indexer

        # Vendor
        indexer.vendor = "EDAX"
        assert not _indexer_is_compatible_with_kikuchipy(indexer, (60, 60), 9)
        with pytest.raises(ValueError, match="`indexer.vendor` must be 'kikuchipy', "):
            _indexer_is_compatible_with_kikuchipy(indexer, (60, 60), raise_if_not=True)
        indexer.vendor = "kikuchipy"

        # Signal shape
        assert not _indexer_is_compatible_with_kikuchipy(indexer, (60, 59), 9)
        with pytest.raises(ValueError, match=r"Indexer signal shape \(60, 60\) must "):
            _indexer_is_compatible_with_kikuchipy(indexer, (60, 59), raise_if_not=True)
        det2 = self.signal.detector.deepcopy()
        det2.shape = (60, 59)
        indexer2 = det2.get_indexer(self.signal.xmap.phases)
        assert not _indexer_is_compatible_with_kikuchipy(indexer2, (60, 60), 9)
        with pytest.raises(ValueError, match=r"Indexer signal shape \(60, 59\) must "):
            _indexer_is_compatible_with_kikuchipy(indexer2, (60, 60), raise_if_not=True)

        # PC
        assert not _indexer_is_compatible_with_kikuchipy(indexer, (60, 60))
        with pytest.raises(
            ValueError, match=r"`indexer.PC` must be an array of shape \(3,\), but was "
        ):
            _indexer_is_compatible_with_kikuchipy(indexer, (60, 60), raise_if_not=True)
        assert not _indexer_is_compatible_with_kikuchipy(indexer, (60, 60), 8)
        with pytest.raises(
            ValueError,
            match=r"`indexer.PC` must be an array of shape \(3,\) or \(8, 3\), but was ",
        ):
            _indexer_is_compatible_with_kikuchipy(
                indexer, (60, 60), 8, raise_if_not=True
            )

    def test_hough_indexing_get_xmap_from_index_data(self):
        phase_list = self.signal.xmap.phases
        xmap, index_data = self.signal.hough_indexing(
            phase_list, self.indexer, return_index_data=True
        )

        with pytest.raises(ValueError, match="`nav_shape` cannot be a tuple of more "):
            _ = kp.indexing.xmap_from_hough_indexing_data(
                index_data, phase_list, navigation_shape=(1, 2, 3)
            )
        with pytest.raises(ValueError, match="`nav_shape` cannot be a tuple of more "):
            _ = kp.indexing.xmap_from_hough_indexing_data(
                index_data, phase_list, navigation_shape=(1.0, 2)
            )

        bad_phase_list = PhaseList(ids=2, space_groups=225)
        with pytest.raises(ValueError, match=r"`phase_list` IDs \[2\] must contain 0"):
            _ = kp.indexing.xmap_from_hough_indexing_data(
                index_data,
                bad_phase_list,
                data_index=0,
            )

        index_data["phase"][-1][0] = -1  # Not indexed
        xmap2 = kp.indexing.xmap_from_hough_indexing_data(
            index_data,
            phase_list,
            navigation_shape=(3, 3),
            step_sizes=(2, 3),
            scan_unit="um",
        )
        assert np.allclose(xmap2.phase_id, [-1] + [0] * 8)
        assert xmap2.scan_unit == "um"
        assert xmap2.dy == 2
        assert xmap2.dx == 3

    def test_reflector_list(self):
        phase_list = self.signal.xmap.phases

        hkl = [[2, 0, 0], [2, 2, 0], [1, 1, 1], [3, 1, 1], [3, 3, 1]]

        ref = ReciprocalLatticeVector(hkl=hkl, phase=phase_list[0])
        reflectors = [hkl, tuple(hkl), np.array(hkl), ref]

        indexers = [self.signal.detector.get_indexer(phase_list)]
        for i in range(len(reflectors)):
            indexer = self.signal.detector.get_indexer(
                phase_list, reflectors=reflectors[i]
            )
            indexers.append(indexer)

            assert np.allclose(indexer.phaselist[0].polefamilies, hkl)

    def test_reflector_lists(self):
        phase_list = PhaseList(names=["a", "b"], space_groups=[186, 225])
        hkl = [
            [[0, 0, 6], [0, 0, -6], [1, 0, 0], [2, 0, 0]],
            [[2, 0, 0], [2, 2, 0]],
            [[1, 1, 1], [3, 1, 1], [3, 3, 1]],
        ]

        _ = self.signal.detector.get_indexer(phase_list, hkl[:2])
        _ = self.signal.detector.get_indexer(phase_list, [hkl[0], None])
        _ = self.signal.detector.get_indexer(phase_list, [None, None])

        with pytest.raises(ValueError, match="One set of reflectors or None must be "):
            _ = self.signal.detector.get_indexer(phase_list, hkl)

    def test_compatible_phase_lists(self):
        phase_list = PhaseList(
            names=["a", "b"],
            space_groups=[186, 225],
            structures=[
                Structure(lattice=Lattice(1, 1, 2, 90, 90, 120)),
                Structure(lattice=Lattice(1, 1, 1, 90, 90, 90)),
            ],
        )
        indexer = self.signal.detector.get_indexer(phase_list)

        assert _phase_lists_are_compatible(phase_list, indexer, True)

        # Differing number of phases
        phase_list2 = phase_list.deepcopy()
        phase_list2.add(Phase("c", space_group=1))
        assert not _phase_lists_are_compatible(phase_list2, indexer)
        with pytest.raises(
            ValueError, match=r"`phase_list` \(3\) and `indexer.phaselist` \(2\)"
        ):
            _ = _phase_lists_are_compatible(phase_list2, indexer, True)

        # Differing lattice parameters
        phase_list3 = phase_list.deepcopy()
        lat = phase_list3["a"].structure.lattice
        lat.setLatPar(lat.a * 10)
        with pytest.raises(
            ValueError, match="Phase 'a' in `phase_list` and phase number 0 in "
        ):
            _ = _phase_lists_are_compatible(phase_list3, indexer, True)

        # Differing space groups
        phase_list4 = phase_list.deepcopy()
        phase_list4["b"].space_group = 224
        with pytest.raises(
            ValueError, match="Phase 'b' in `phase_list` and phase number 1 in "
        ):
            _ = _phase_lists_are_compatible(phase_list4, indexer, True)


@pytest.mark.skipif(
    not kp._pyebsdindex_installed, reason="pyebsdindex is not installed"
)
class TestPCOptimization:
    def setup_method(self):
        s = kp.data.nickel_ebsd_small()
        s.remove_static_background()
        s.remove_dynamic_background()

        self.signal = s
        self.indexer = s.detector.get_indexer(s.xmap.phases)

    def test_optimize_pc(self):
        det0 = self.signal.detector

        det = self.signal.hough_indexing_optimize_pc(det0.pc_average, self.indexer)
        assert det.navigation_shape == (1,)
        assert np.allclose(det.pc_average, det0.pc_average, atol=1e-2)

        # Batch with PC array with more than one dimension
        det2 = self.signal.hough_indexing_optimize_pc(
            det0.pc_average, self.indexer, batch=True
        )
        assert det2.navigation_shape == (3, 3)
        assert np.allclose(det.pc_average, det2.pc_average, atol=1e-2)

        # Detector parameters
        assert det.shape == det0.shape
        assert np.isclose(det.sample_tilt, det0.sample_tilt)
        assert np.isclose(det.tilt, det0.tilt)
        assert np.isclose(det.px_size, det0.px_size)

    def test_optimize_pc_pso(self, worker_id):
        det0 = self.signal.detector

        det = self.signal.hough_indexing_optimize_pc(
            det0.pc_average, self.indexer, method="PSO"
        )
        # Results are not deterministic, so we give a wide range here...
        assert abs(det0.pc_average - det.pc_average).max() < 0.03

        if worker_id == "master":  # pragma: no cover
            # Batch with PC array with more than one dimension
            det2 = self.signal.hough_indexing_optimize_pc(
                det0.pc_average,
                self.indexer,
                batch=True,
                method="PSO",
                search_limit=0.1,
            )
            assert det2.navigation_shape == (3, 3)
            assert abs(det.pc_average - det2.pc_average).max() < 0.03

    def test_optimize_pc_raises(self):
        with pytest.raises(ValueError, match="`pc0` must be of size 3"):
            _ = self.signal.hough_indexing_optimize_pc([0.5, 0.5], self.indexer)

        with pytest.raises(ValueError, match="`method` 'powell' must be one of the "):
            _ = self.signal.hough_indexing_optimize_pc(
                [0.5, 0.5, 0.5], self.indexer, method="Powell"
            )

    def test_optimize_pc_lazy(self):  # pragma: no cover
        s = self.signal.as_lazy()
        det = self.signal.detector

        if kp._pyopencl_context_available:
            det = s.hough_indexing_optimize_pc(det.pc_average, self.indexer)
            assert np.allclose(det.pc_average, det.pc_average, atol=1e-2)
        else:
            with pytest.raises(ValueError, match="Hough indexing of lazy signals must"):
                _ = s.hough_indexing_optimize_pc(det.pc_average, self.indexer)


@pytest.mark.skipif(kp._pyebsdindex_installed, reason="pyebsdindex is installed")
class TestHoughIndexingNoPyEBSDIndex:  # pragma: no cover
    def setup_method(self):
        s = kp.data.nickel_ebsd_small()

        self.signal = s

    def test_get_indexer(self):
        with pytest.raises(ValueError, match="pyebsdindex must be installed"):
            _ = self.signal.detector.get_indexer(None)

    def test_hough_indexing_raises_pyebsdindex(self):
        with pytest.raises(ValueError, match="pyebsdindex to be installed"):
            _ = self.signal.hough_indexing(None, None)

    def test_optimize_pc_raises_pyebsdindex(self):
        with pytest.raises(ValueError, match="pyebsdindex to be installed"):
            _ = self.signal.hough_indexing_optimize_pc(None, None)

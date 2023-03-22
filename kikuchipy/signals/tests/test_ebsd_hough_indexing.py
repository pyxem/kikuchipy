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

import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import CrystalMap, PhaseList
import pytest

import kikuchipy as kp
from kikuchipy.indexing._hough_indexing import (
    _get_info_message,
    _indexer_is_compatible_with_kikuchipy,
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
        self.xmap = s.xmap
        self.phase_list = s.xmap.phases
        self.detector = s.detector
        self.indexer = s.detector.get_indexer(s.xmap.phases)

    def test_hough_indexing_print_information(self):
        det = self.detector
        det.shape = (4, 5)  # Save time on indexer creation

        indexer1 = det.get_indexer(self.phase_list)

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
        indexer2 = det_pc_mean.get_indexer(self.phase_list)
        info2 = _get_info_message(
            self.signal.axes_manager.navigation_size, chunksize=3, indexer=indexer2
        )
        info_list2 = info2.split("\n")
        assert info_list2[2] == "  Projection center (Bruker): (0.4251, 0.2134, 0.5007)"

    def test_hough_indexing(self):
        # Reference results (Hough indexing + refinement)
        xmap_ref = self.xmap

        phase_list = self.phase_list
        indexer = self.indexer

        xmap = self.signal.hough_indexing(phase_list, indexer)

        assert isinstance(xmap, CrystalMap)
        assert xmap.phases.names == phase_list.names
        angles = xmap.orientations.angle_with(xmap_ref.orientations, degrees=True)
        assert np.all(angles < 1)

    def test_hough_indexing_plot_transform(self):
        _ = self.signal.hough_indexing(self.phase_list, self.indexer, verbose=2)
        ax = plt.gca()
        assert len(ax.texts) == 9
        for i, text in enumerate(ax.texts):
            assert text.get_text() == str(i + 1)

    def test_hough_indexing_lazy(self):  # pragma: no cover
        s = self.signal.as_lazy()

        if kp._pyopencl_context_available:
            xmap1 = s.hough_indexing(self.phase_list, self.indexer)
            xmap2 = self.signal.hough_indexing(self.phase_list, self.indexer)
            assert np.allclose(xmap1.rotations.data, xmap2.rotations.data)
            assert np.allclose(xmap1.fit, xmap2.fit)
        else:
            with pytest.raises(ValueError, match="Hough indexing of lazy signals must"):
                _ = s.hough_indexing(self.phase_list, self.indexer, verbose=2)

    def test_hough_indexing_return_index_data(self):
        phase_list = self.phase_list
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
        indexer = self.detector.get_indexer(self.phase_list, nBands=8)
        _, band_data = self.signal.hough_indexing(
            self.phase_list, indexer, return_band_data=True
        )
        assert isinstance(band_data, np.ndarray)
        assert band_data.shape == (
            self.signal.axes_manager.navigation_size,
            indexer.bandDetectPlan.nBands,
        )

        _, index_data, band_data = self.signal.hough_indexing(
            self.phase_list, self.indexer, return_index_data=True, return_band_data=True
        )
        assert isinstance(index_data, np.ndarray)
        assert isinstance(band_data, np.ndarray)
        assert index_data.shape == (2, 9)
        assert band_data.shape == (9, 9)

    def test_hough_indexing_raises_dissimilar_phase_lists(self):
        phase_list = PhaseList(names=["a", "b"], space_groups=[225, 229])
        with pytest.raises(ValueError, match=r"`indexer.phaselist` \['FCC'\] and the "):
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
        det2 = self.detector.deepcopy()
        det2.shape = (60, 59)
        indexer2 = det2.get_indexer(self.phase_list)
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

        # Phase list
        indexer.phaselist = ["FCC", "FCC"]
        assert not _indexer_is_compatible_with_kikuchipy(indexer, (60, 60), 9)
        with pytest.raises(
            ValueError,
            match=r"`indexer.phaselist` must be one of \[\['FCC'\], \['BCC'\], \['FCC',",
        ):
            _indexer_is_compatible_with_kikuchipy(
                indexer, (60, 60), 9, raise_if_not=True
            )

    def test_hough_indexing_get_xmap_from_index_data(self):
        phase_list = self.phase_list
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

    def test_optimize_pc(self):
        # Batch
        det2 = self.signal.hough_indexing_optimize_pc(
            self.detector.pc_average, self.indexer
        )
        assert det2.navigation_shape == (1,)
        assert np.allclose(det2.pc_average, self.detector.pc_average, atol=1e-2)
        det3 = self.signal.hough_indexing_optimize_pc(
            self.detector.pc_average, self.indexer, batch=True
        )
        assert det3.navigation_shape == (3, 3)
        assert np.allclose(det2.pc_average, det3.pc_average, atol=1e-2)

        # Detector parameters
        assert det2.shape == self.detector.shape
        assert np.isclose(det2.sample_tilt, self.detector.sample_tilt)
        assert np.isclose(det2.tilt, self.detector.tilt)
        assert np.isclose(det2.px_size, self.detector.px_size)

    def test_optimize_pc_pso(self):
        det = self.signal.hough_indexing_optimize_pc(
            self.detector.pc_average, self.indexer, method="PSO"
        )
        # Results are not deterministic, so we give a wide range here...
        assert abs(self.detector.pc_average - det.pc_average).max() < 0.05

    def test_optimize_pc_raises(self):
        with pytest.raises(ValueError, match="`pc0` must be of size 3"):
            _ = self.signal.hough_indexing_optimize_pc([0.5, 0.5], self.indexer)

        with pytest.raises(ValueError, match="`method` 'powell' must be one of the "):
            _ = self.signal.hough_indexing_optimize_pc(
                [0.5, 0.5, 0.5], self.indexer, method="Powell"
            )

        with pytest.raises(ValueError, match="PSO optimization method does not "):
            _ = self.signal.hough_indexing_optimize_pc(
                [0.5, 0.5, 0.5], self.indexer, method="PSO", batch=True
            )

    def test_optimize_pc_lazy(self):  # pragma: no cover
        s = self.signal.as_lazy()

        if kp._pyopencl_context_available:
            det = s.hough_indexing_optimize_pc(self.detector.pc_average, self.indexer)
            assert np.allclose(det.pc_average, self.detector.pc_average, atol=1e-2)
        else:
            with pytest.raises(ValueError, match="Hough indexing of lazy signals must"):
                _ = s.hough_indexing_optimize_pc(self.detector.pc_average, self.indexer)


@pytest.mark.skipif(kp._pyebsdindex_installed, reason="pyebsdindex is installed")
class TestHoughIndexingNopyebsdindex:  # pragma: no cover
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

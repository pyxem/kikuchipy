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
from kikuchipy.indexing._hough_indexing import _get_info_message


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

    @pytest.mark.skipif(
        not kp._pyebsdindex_installed, reason="pyebsdindex is not installed"
    )
    def test_hough_indexing_print_information(self):
        det = self.detector
        det.shape = (4, 5)  # Save time on indexer creation

        indexer1 = det.get_indexer(self.phase_list)

        info = _get_info_message(
            self.signal.axes_manager.navigation_size, chunksize=3, indexer=indexer1
        )
        info_list = info.split("\n")
        assert info_list[0] == "Hough indexing with PyEBSDIndex information:"
        assert info_list[1][:7] == "  GPU: "  # GPU not available on test machines
        assert info_list[2] == "  Projection center (mean): (0.4251, 0.2134, 0.5007)"
        assert info_list[3] == "  Indexing 9 pattern(s) in 3 chunk(s)"

        det_pc_mean = det.deepcopy()
        det_pc_mean.pc = det_pc_mean.pc_average
        indexer2 = det_pc_mean.get_indexer(self.phase_list)
        info2 = _get_info_message(
            self.signal.axes_manager.navigation_size, chunksize=3, indexer=indexer2
        )
        info_list2 = info2.split("\n")
        assert info_list2[2] == "  Projection center: (0.4251, 0.2134, 0.5007)"

    @pytest.mark.skipif(
        not kp._pyebsdindex_installed, reason="pyebsdindex is not installed"
    )
    def test_hough_indexing(self):
        # Reference results (Hough indexing + refinement)
        xmap_ref = self.xmap

        phase_list = self.phase_list
        indexer = self.indexer

        xmap = self.signal.hough_indexing(phase_list, indexer)

        assert isinstance(xmap, CrystalMap)
        assert xmap.phases.names == phase_list.names
        angles = np.rad2deg(xmap.orientations.angle_with(xmap_ref.orientations))
        assert np.all(angles < 1)

    @pytest.mark.skipif(
        not kp._pyebsdindex_installed, reason="pyebsdindex is not installed"
    )
    def test_hough_indexing_plot_transform(self):
        _ = self.signal.hough_indexing(self.phase_list, self.indexer, verbose=2)
        ax = plt.gca()
        assert len(ax.texts) == 9
        for i, text in enumerate(ax.texts):
            assert text.get_text() == str(i + 1)

    @pytest.mark.skipif(
        not kp._pyebsdindex_installed, reason="pyebsdindex is not installed"
    )
    def test_hough_indexing_lazy(self):
        s = self.signal.as_lazy()

        from pyebsdindex import _pyopencl_installed

        if not _pyopencl_installed:
            with pytest.raises(ValueError, match="Hough indexing of lazy signals must"):
                _ = s.hough_indexing(self.phase_list, self.indexer, verbose=2)
        else:  # pragma: no cover
            xmap1 = s.hough_indexing(self.phase_list, self.indexer)
            xmap2 = self.signal.hough_indexing(self.phase_list, self.indexer)
            assert np.allclose(xmap1.rotations.data, xmap2.rotations.data)
            assert np.allclose(xmap1.fit, xmap2.fit)

    @pytest.mark.skipif(
        not kp._pyebsdindex_installed, reason="pyebsdindex is not installed"
    )
    def test_hough_indexing_return_index_data(self):
        phase_list = self.phase_list
        xmap, index_data = self.signal.hough_indexing(
            phase_list, self.indexer, return_index_data=True
        )
        assert list(index_data.dtype.fields.keys()) == [
            "quat",
            "iq",
            "pq",
            "cm",
            "phase",
            "fit",
            "nmatch",
            "matchattempts",
            "totvotes",
        ]
        assert index_data.shape == (2, 9)
        xmap3 = kp.indexing.xmap_from_hough_indexing_data(index_data, phase_list)
        assert xmap3.shape == (9,)
        assert np.allclose(xmap3.rotations.data, xmap.rotations.data)
        assert np.allclose(xmap3.fit, xmap.fit)
        assert np.allclose(xmap3.cm, xmap.cm)

    @pytest.mark.skipif(
        not kp._pyebsdindex_installed, reason="pyebsdindex is not installed"
    )
    def test_hough_indexing_raises_dissimilar_phase_lists(self):
        phase_list = PhaseList(names=["a", "b"], space_groups=[225, 229])
        with pytest.raises(ValueError, match=r"`indexer.phaselist` \['FCC'\] and the "):
            _ = self.signal.hough_indexing(phase_list, self.indexer)

    @pytest.mark.skipif(kp._pyebsdindex_installed, reason="pyebsdindex is installed")
    def test_hough_indexing_raises_pyebsdindex(self):
        with pytest.raises(ValueError, match="Hough indexing requires pyebsdindex to "):
            _ = self.signal.hough_indexing(self.phase_list, self.indexer)

    @pytest.mark.skipif(
        not kp._pyebsdindex_installed, reason="pyebsdindex is not installed"
    )
    def test_indexer_is_compatible_with_signal(self):
        pass

    @pytest.mark.skipif(
        not kp._pyebsdindex_installed, reason="pyebsdindex is not installed"
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


class TestOptimizePC:
    @pytest.mark.skipif(
        not kp._pyebsdindex_installed, reason="pyebsdindex is not installed"
    )
    def test_optimize_pc(self):
        pass

    @pytest.mark.skipif(
        not kp._pyebsdindex_installed, reason="pyebsdindex is not installed"
    )
    def test_optimize_pc_raises(self):
        pass

    def test_optimize_pc_raises_pyebsdindex(self):
        pass

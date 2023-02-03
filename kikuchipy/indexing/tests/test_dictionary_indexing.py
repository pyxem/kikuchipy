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

import dask.array as da
import numpy as np
from orix.crystal_map import CrystalMap
import pytest

import kikuchipy as kp


class TestDictionaryIndexing:
    def test_dictionary_indexing_doesnt_change_data(self, dummy_signal):
        """Scores are all 1 for a dictionary containing all patterns
        from dummy_signal().
        """
        s_dict = kp.signals.EBSD(dummy_signal.data.reshape(-1, 3, 3))
        s_dict.axes_manager[0].name = "x"
        s_dict.xmap = CrystalMap.empty((9,))
        dummy_signal2 = dummy_signal.deepcopy()
        s_dict2 = s_dict.deepcopy()
        xmap = dummy_signal2.dictionary_indexing(s_dict2, metric="ndp", rechunk=True)

        assert isinstance(xmap, CrystalMap)
        assert np.allclose(xmap.scores[:, 0], 1)

        # Data is not affected by indexing method
        assert np.allclose(dummy_signal.data, dummy_signal2.data)
        assert np.allclose(s_dict.data, s_dict2.data)

    def test_dictionary_indexing_signal_mask(self, dummy_signal):
        """Passing a signal mask works, using 64-bit floats works,
        rechunking of experimental patterns works.
        """
        s_dict = kp.signals.EBSD(dummy_signal.data.reshape(-1, 3, 3))
        s_dict.axes_manager[0].name = "x"
        s_dict.xmap = CrystalMap.empty((9,))
        signal_mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
        xmap = dummy_signal.dictionary_indexing(
            s_dict,
            dtype=np.float64,
            n_per_iteration=2,
            signal_mask=signal_mask,
            rechunk=True,
        )
        assert np.allclose(xmap.scores[:, 0], 1)

        # Raises
        with pytest.raises(ValueError, match="The signal mask must be a NumPy array"):
            _ = dummy_signal.dictionary_indexing(
                s_dict, signal_mask=da.from_array(signal_mask)
            )

    def test_dictionary_indexing_n_per_iteration_from_lazy(self, dummy_signal):
        """Getting number of iterations from Dask array chunk works, and
        NDP rechunking of experimental patterns works.
        """
        s_dict = kp.signals.EBSD(dummy_signal.data.reshape(-1, 3, 3))
        s_dict.axes_manager[0].name = "x"
        s_dict.xmap = CrystalMap.empty((9,))
        s_dict_lazy = s_dict.as_lazy()
        s_dict_lazy.xmap = s_dict.xmap
        signal_mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
        xmap = dummy_signal.dictionary_indexing(
            s_dict_lazy, metric="ndp", signal_mask=signal_mask
        )
        assert np.allclose(xmap.scores[:, 0], 1)

        # So that computing parts of the dictionary during indexing is
        # covered, and NDP rechunking
        xmap4 = dummy_signal.dictionary_indexing(
            s_dict_lazy, metric="ndp", n_per_iteration=2, rechunk=True
        )
        assert np.allclose(xmap4.scores[:, 0], 1)

    def test_dictionary_indexing_invalid_metric(self, dummy_signal):
        s_dict = kp.signals.EBSD(dummy_signal.data.reshape(-1, 3, 3))
        s_dict.axes_manager[0].name = "x"
        s_dict.xmap = CrystalMap.empty((9,))
        with pytest.raises(ValueError, match="'invalid' must be either of "):
            _ = dummy_signal.dictionary_indexing(s_dict, metric="invalid")

    def test_dictionary_indexing_invalid_signal_shapes(self, dummy_signal):
        s_dict_data = dummy_signal.data[:, :, :2, :2].reshape((-1, 2, 2))
        s_dict = kp.signals.EBSD(s_dict_data)
        s_dict.axes_manager[0].name = "x"
        s_dict.xmap = CrystalMap.empty((9,))
        with pytest.raises(ValueError):
            _ = dummy_signal.dictionary_indexing(s_dict)

    def test_dictionary_indexing_invalid_dictionary(self, dummy_signal):
        s_dict = kp.signals.EBSD(dummy_signal.data)
        s_dict.axes_manager[0].name = "x"
        s_dict.axes_manager[1].name = "y"

        # Dictionary xmap property is empty
        with pytest.raises(ValueError, match="Dictionary signal must have a non-empty"):
            _ = dummy_signal.dictionary_indexing(s_dict)

        # Dictionary not 1 navigation dimension
        s_dict.xmap = CrystalMap.empty((3, 3))
        with pytest.raises(ValueError, match="Dictionary signal must have a non-empty"):
            _ = dummy_signal.dictionary_indexing(s_dict)

    @pytest.mark.parametrize(
        "nav_slice, nav_shape, unit",
        [
            ((0, 0), (), "px"),  # 0D
            ((0, slice(0, 1)), (), "um"),  # 0D
            ((0, slice(0, 3)), (3,), "um"),  # 1D
            ((slice(0, 3), slice(0, 2)), (2, 3), "um"),  # 2D
        ],
    )
    def test_dictionary_indexing_nav_shape(
        self, dummy_signal, nav_slice, nav_shape, unit
    ):
        """Dictionary indexing handles experimental datasets of all
        allowed navigation shapes of 0D, 1D and 2D.
        """
        s = dummy_signal.inav[nav_slice]
        for ax in s.axes_manager.navigation_axes:
            ax.units = "um"
        s_dict = kp.signals.EBSD(dummy_signal.data.reshape(-1, 3, 3))
        s_dict.axes_manager[0].name = "x"
        dict_size = s_dict.axes_manager.navigation_size
        s_dict.xmap = CrystalMap.empty((dict_size,))
        xmap = s.dictionary_indexing(s_dict)
        assert xmap.shape == nav_shape
        assert np.allclose(xmap.scores[:, 0], np.ones(int(np.prod(nav_shape))))
        assert xmap.scan_unit == unit

    def test_dictionary_indexing_navigation_mask_raises(self, dummy_signal):
        s = dummy_signal
        s_dict = kp.signals.EBSD(dummy_signal.data.reshape(-1, 3, 3))
        s_dict.axes_manager[0].name = "x"
        s_dict.xmap = CrystalMap.empty((s_dict.axes_manager.navigation_size,))

        nav_mask1 = np.ones(8, dtype=bool)
        with pytest.raises(ValueError, match=r"The navigation mask shape \(8,\) and "):
            _ = s.dictionary_indexing(s_dict, navigation_mask=nav_mask1)

        nav_mask2 = np.ones((3, 3), dtype=bool)
        with pytest.raises(ValueError, match=r"The navigation mask must allow for "):
            _ = s.dictionary_indexing(s_dict, navigation_mask=nav_mask2)

        nav_mask3 = da.ones((3, 3), dtype=bool)
        nav_mask3[0, 0] = False
        with pytest.raises(ValueError, match=r"The navigation mask must be a NumPy "):
            _ = s.dictionary_indexing(s_dict, navigation_mask=nav_mask3)

    def test_dictionary_indexing_navigation_mask(self, dummy_signal):
        s = dummy_signal
        nav_shape = s._navigation_shape_rc
        nav_size = int(np.prod(nav_shape))
        s_dict = kp.signals.EBSD(dummy_signal.data.reshape(nav_size, 3, 3))
        s_dict.axes_manager[0].name = "x"
        s_dict.xmap = CrystalMap.empty((nav_size,))

        nav_mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
        xmap1 = s.dictionary_indexing(s_dict, keep_n=1, navigation_mask=nav_mask)
        xmap2 = s.dictionary_indexing(s_dict, metric="ndp", navigation_mask=~nav_mask)
        assert xmap1.size == 8
        assert xmap1.rotations_per_point == 1
        assert xmap2.size == 1
        assert xmap2.rotations_per_point == s_dict.xmap.size

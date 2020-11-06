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

import numpy as np
import pytest

from kikuchipy.indexing._merge_crystal_maps import merge_crystal_maps
from kikuchipy.indexing.similarity_metrics import make_similarity_metric


class TestMergeCrystalMaps:
    @pytest.mark.parametrize(
        "map_shape, n_rot_per_point, phase_names",
        [
            ((0, 3), 10, ["a", "b"]),
            ((0, 4), 1, ["a", "b", "c"]),
            ((3, 0), 5, ["austenite", "ferrite"]),
            ((4, 0), 1, ["al", "cu", "si"]),
        ],
    )
    def test_merge_crystal_maps_1d(
        self, get_single_phase_xmap, map_shape, n_rot_per_point, phase_names
    ):
        """Crystal maps with a 1D navigation shape can be merged
        successfully and yields an expected output.
        """
        n_phases = len(phase_names)
        scores_prop, sim_idx_prop = "scores", "sim_idx"

        map_size = np.sum(map_shape)
        data_shape = (map_size,)
        if n_rot_per_point > 1:
            data_shape += (n_rot_per_point,)

        desired_phase_ids = np.zeros(map_size)
        desired_scores = np.ones(data_shape)
        desired_idx = np.arange(np.prod(data_shape)).reshape(data_shape)

        xmaps = []
        xmap_args = (map_shape, n_rot_per_point, [scores_prop, sim_idx_prop])
        phase_ids = np.arange(n_phases)
        for i in range(n_phases):
            xmap = get_single_phase_xmap(
                *xmap_args, phase_names[i], phase_ids[i]
            )
            # All maps have at least one point with the best score
            xmap[i].prop[scores_prop] += i + 1
            xmaps.append(xmap)

            desired_phase_ids[i] = i
            desired_scores[i] = xmap[i].prop[scores_prop]
            desired_idx[i] = xmap[i].prop[sim_idx_prop]

            if i == 0:
                desired_rot = xmap.rotations.data
            else:
                desired_rot[i] = xmap[i].rotations.data

        merged_xmap = merge_crystal_maps(
            crystal_maps=xmaps,
            scores_prop=scores_prop,
            simulation_indices_prop=sim_idx_prop,
        )

        assert merged_xmap.shape == xmaps[0].shape
        assert merged_xmap.size == xmaps[0].size
        for v1, v2 in zip(
            merged_xmap._coordinates.values(), xmaps[0]._coordinates.values()
        ):
            if v1 is None:
                assert v1 is v2
            else:
                np.allclose(v1, v2)

        assert np.allclose(merged_xmap.phase_id, desired_phase_ids)
        assert np.allclose(merged_xmap.prop[scores_prop], desired_scores)
        assert np.allclose(merged_xmap.prop[sim_idx_prop], desired_idx)
        assert np.allclose(merged_xmap.rotations.data, desired_rot)

    @pytest.mark.parametrize(
        "map_shape, n_rot_per_point, phase_names",
        [
            ((4, 3), 10, ["a", "b"]),
            ((5, 4), 1, ["a", "b", "c"]),
            ((3, 4), 5, ["austenite", "ferrite"]),
            ((4, 5), 1, ["al", "cu", "si"]),
        ],
    )
    def test_merge_crystal_maps_2d(
        self, get_single_phase_xmap, map_shape, n_rot_per_point, phase_names
    ):
        """Crystal maps with a 2D navigation shape can be merged
        successfully and yields an expected output.
        """
        n_phases = len(phase_names)
        scores_prop, sim_idx_prop = "scores", "sim_idx"

        map_size = np.prod(map_shape)
        data_shape = (map_size,)
        if n_rot_per_point > 1:
            data_shape += (n_rot_per_point,)

        desired_phase_ids = np.zeros(map_size)
        desired_scores = np.ones(data_shape)
        desired_idx = np.arange(np.prod(data_shape)).reshape(data_shape)

        xmaps = []
        xmap_args = (map_shape, n_rot_per_point, [scores_prop, sim_idx_prop])
        phase_ids = np.arange(n_phases)
        ny, nx = map_shape
        for i in range(n_phases):
            xmap = get_single_phase_xmap(
                *xmap_args, phase_names[i], phase_ids[i]
            )
            # All maps have at least one point with the best score along
            # the map diagonal
            idx = (i, i)
            xmap[idx].prop[scores_prop] += i + 1
            xmaps.append(xmap)

            j = i * (1 + nx)
            desired_phase_ids[j] = i
            desired_scores[j] = xmap[idx].prop[scores_prop]
            desired_idx[j] = xmap[idx].prop[sim_idx_prop]

            if i == 0:
                desired_rot = xmap.rotations.data
            else:
                desired_rot[j] = xmap[idx].rotations.data

        merged_xmap = merge_crystal_maps(
            crystal_maps=xmaps,
            scores_prop=scores_prop,
            simulation_indices_prop=sim_idx_prop,
        )

        assert merged_xmap.shape == xmaps[0].shape
        assert merged_xmap.size == xmaps[0].size
        for v1, v2 in zip(
            merged_xmap._coordinates.values(), xmaps[0]._coordinates.values()
        ):
            if v1 is None:
                assert v1 is v2
            else:
                np.allclose(v1, v2)

        assert np.allclose(merged_xmap.phase_id, desired_phase_ids)
        assert np.allclose(merged_xmap.prop[scores_prop], desired_scores)
        assert np.allclose(merged_xmap.prop[sim_idx_prop], desired_idx)
        assert np.allclose(merged_xmap.rotations.data, desired_rot)

    def test_more_indices_than_scores_per_point(self):
        pass

    def test_more_scores_than_indices_per_point(self):
        pass

    def test_merging_maps_different_shapes_raises(self):
        pass

    def test_merging_maps_different_number_of_scores(self):
        pass

    @pytest.mark.parametrize(
        "scores_prop, sim_idx_prop",
        [("scores", "sim_idx"), ("similar", "simulated")],
    )
    def test_property_names(
        self, get_single_phase_xmap, scores_prop, sim_idx_prop
    ):
        """Passing scores and simulation indices property names returns
        expected properties in merged map.
        """
        map_shape = (5, 6)
        n_rot_per_point = 50

        xmap1 = get_single_phase_xmap(
            map_shape, n_rot_per_point, [scores_prop, sim_idx_prop], "a", 0
        )
        xmap2 = get_single_phase_xmap(
            map_shape, n_rot_per_point, [scores_prop, sim_idx_prop], "b", 1
        )

        xmap2[3, 3].prop[scores_prop] = 2
        merged_xmap = merge_crystal_maps(
            crystal_maps=[xmap1, xmap2],
            scores_prop=scores_prop,
            simulation_indices_prop=sim_idx_prop,
        )

        assert scores_prop in merged_xmap.prop.keys()
        assert sim_idx_prop in merged_xmap.prop.keys()

    def test_negative_metric(self, get_single_phase_xmap):
        def negative_sad(p, t):  # pragma: no cover
            return -np.sum(np.abs(p - t), axis=(2, 3))

        metric = make_similarity_metric(negative_sad, greater_is_better=False)

        map_shape = (5, 6)
        n_rot_per_point = 5
        scores_prop = "scores"
        sim_idx_prop = "simulation_indices"

        xmap1 = get_single_phase_xmap(
            map_shape, n_rot_per_point, [scores_prop, sim_idx_prop], "a", 0
        )
        xmap2 = get_single_phase_xmap(
            map_shape, n_rot_per_point, [scores_prop, sim_idx_prop], "b", 1
        )

        xmap2[0, 3].prop[scores_prop] = 0
        desired_phase_id = np.zeros(np.prod(map_shape))
        desired_phase_id[3] = 1

        merged_xmap = merge_crystal_maps(
            crystal_maps=[xmap1, xmap2], metric=metric,
        )

        assert np.allclose(merged_xmap.phase_id, desired_phase_id)

    def test_mean_n_best(self):
        pass

    def test_merging_returns_same_map(self):
        pass

    @pytest.mark.parametrize(
        "phase_names, desired_phase_names",
        [
            (["a"] * 3, ["a", "a1", "a2"]),
            (["hello_there1"] * 2, ["hello_there1", "hello_there11"]),
            (["1"] * 5, ["1", "11", "12", "13", "14"]),
        ],
    )
    def test_warning_merge_maps_with_same_phase(
        self, get_single_phase_xmap, phase_names, desired_phase_names,
    ):
        n_phases = len(phase_names)
        scores_prop = "scores"
        sim_idx_prop = "simulated_indices"
        map_shape = (5, 6)
        n_rot_per_point = 5

        xmaps = []
        xmap_args = (map_shape, n_rot_per_point, [scores_prop, sim_idx_prop])
        phase_ids = np.arange(n_phases)
        for i in range(n_phases):
            xmap = get_single_phase_xmap(
                *xmap_args, phase_names[i], phase_ids[i]
            )
            # All maps have at least one point with the best score
            xmap[i, i].scores += i + 1
            xmaps.append(xmap)

        with pytest.warns(
            UserWarning, match=f"There are duplicates of phase {phase_names[0]}"
        ):
            merged_xmap = merge_crystal_maps(
                crystal_maps=xmaps,
                scores_prop=scores_prop,
                simulation_indices_prop=sim_idx_prop,
            )

        assert all(
            [name in merged_xmap.phases.names for name in desired_phase_names]
        )

    def test_combined_scores_shape(self):
        pass

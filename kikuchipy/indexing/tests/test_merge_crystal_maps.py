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
        "map_shape, rot_per_point, phase_names",
        [
            ((0, 3), 10, ["a", "b"]),
            ((0, 4), 1, ["a", "b", "c"]),
            ((3, 0), 5, ["austenite", "ferrite"]),
            ((4, 0), 1, ["al", "cu", "si"]),
        ],
    )
    def test_merge_crystal_maps_1d(
        self, get_single_phase_xmap, map_shape, rot_per_point, phase_names
    ):
        """Crystal maps with a 1D navigation shape can be merged
        successfully and yields an expected output.
        """
        n_phases = len(phase_names)
        scores_prop, sim_idx_prop = "scores", "sim_idx"

        map_size = np.sum(map_shape)
        data_shape = (map_size,)
        if rot_per_point > 1:
            data_shape += (rot_per_point,)

        desired_phase_ids = np.zeros(map_size)
        desired_scores = np.ones(data_shape)
        desired_idx = np.arange(np.prod(data_shape)).reshape(data_shape)

        xmaps = []
        xmap_args = (map_shape, rot_per_point, [scores_prop, sim_idx_prop])
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

        desired_merged_shapes = (map_size, rot_per_point * n_phases)
        assert (
            merged_xmap.prop[f"merged_{scores_prop}"].shape
            == desired_merged_shapes
        )
        assert (
            merged_xmap.prop[f"merged_{sim_idx_prop}"].shape
            == desired_merged_shapes
        )

    @pytest.mark.parametrize(
        "map_shape, rot_per_point, phase_names, mean_n_best",
        [
            ((4, 3), 10, ["a", "b"], 5),
            ((5, 4), 1, ["a", "b", "c"], 1),
            ((3, 4), 5, ["austenite", "ferrite"], 4),
            ((4, 5), 1, ["al", "cu", "si"], 1),
        ],
    )
    def test_merge_crystal_maps_2d(
        self,
        get_single_phase_xmap,
        map_shape,
        rot_per_point,
        phase_names,
        mean_n_best,
    ):
        """Crystal maps with a 2D navigation shape can be merged
        successfully and yields an expected output.
        """
        n_phases = len(phase_names)
        scores_prop, sim_idx_prop = "scores", "sim_idx"

        map_size = np.prod(map_shape)
        data_shape = (map_size,)
        if rot_per_point > 1:
            data_shape += (rot_per_point,)

        desired_phase_ids = np.zeros(map_size)
        desired_scores = np.ones(data_shape)
        desired_idx = np.arange(np.prod(data_shape)).reshape(data_shape)

        xmaps = []
        xmap_args = (map_shape, rot_per_point, [scores_prop, sim_idx_prop])
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
            mean_n_best=mean_n_best,
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

        desired_merged_shapes = (map_size, rot_per_point * n_phases)
        assert (
            merged_xmap.prop[f"merged_{scores_prop}"].shape
            == desired_merged_shapes
        )
        assert (
            merged_xmap.prop[f"merged_{sim_idx_prop}"].shape
            == desired_merged_shapes
        )

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
        rot_per_point = 50

        xmap1 = get_single_phase_xmap(
            map_shape, rot_per_point, [scores_prop, sim_idx_prop], "a", 0
        )
        xmap2 = get_single_phase_xmap(
            map_shape, rot_per_point, [scores_prop, sim_idx_prop], "b", 1
        )

        xmap2[3, 3].prop[scores_prop] = 2
        merged_xmap = merge_crystal_maps(
            crystal_maps=[xmap1, xmap2],
            scores_prop=scores_prop,
            simulation_indices_prop=sim_idx_prop,
        )

        assert scores_prop in merged_xmap.prop.keys()
        assert sim_idx_prop in merged_xmap.prop.keys()

        desired_merged_shapes = (np.prod(map_shape), rot_per_point * 2)
        assert (
            merged_xmap.prop[f"merged_{scores_prop}"].shape
            == desired_merged_shapes
        )
        assert (
            merged_xmap.prop[f"merged_{sim_idx_prop}"].shape
            == desired_merged_shapes
        )

    def test_negative_metric(self, get_single_phase_xmap):
        def negative_sad(p, t):  # pragma: no cover
            return -np.sum(np.abs(p - t), axis=(2, 3))

        metric = make_similarity_metric(negative_sad, greater_is_better=False)

        map_shape = (5, 6)
        rot_per_point = 5
        scores_prop = "scores"
        sim_idx_prop = "simulation_indices"

        xmap1 = get_single_phase_xmap(
            map_shape, rot_per_point, [scores_prop, sim_idx_prop], "a", 0
        )
        xmap2 = get_single_phase_xmap(
            map_shape, rot_per_point, [scores_prop, sim_idx_prop], "b", 1
        )

        xmap2[0, 3].prop[scores_prop] = 0
        desired_phase_id = np.zeros(np.prod(map_shape))
        desired_phase_id[3] = 1

        merged_xmap = merge_crystal_maps(
            crystal_maps=[xmap1, xmap2], metric=metric,
        )

        assert np.allclose(merged_xmap.phase_id, desired_phase_id)

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
        rot_per_point = 5

        xmaps = []
        xmap_args = (map_shape, rot_per_point, [scores_prop, sim_idx_prop])
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

    @pytest.mark.parametrize(
        (
            "nav_shape, rot_per_point, mean_n_best, desired_merged_scores, "
            "desired_merged_sim_idx"
        ),
        [
            ((2, 0), 1, 1, [[1, 1], [2, 1]], [[0, 2], [3, 1]]),
            ((1, 2), 1, 1, [[1, 1], [2, 1]], [[0, 2], [3, 1]]),
            (
                (1, 3),
                1,
                1,
                [[1, 1, 1], [2, 1, 1], [3, 1, 1]],
                [[0, 3, 6], [4, 1, 7], [8, 2, 5]],
            ),
            (
                (2, 1),
                2,
                2,
                [[1, 1, 1, 1], [2, 2, 1, 1]],
                [[0, 4, 1, 5], [6, 7, 2, 3]],
            ),
            (
                (3, 2),
                1,
                1,
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [2, 1, 1],
                    [2, 1, 1],
                    [3, 1, 1],
                    [3, 1, 1],
                ],
                [
                    [0, 6, 12],
                    [1, 7, 13],
                    [8, 2, 14],
                    [9, 3, 15],
                    [16, 4, 10],
                    [17, 5, 11],
                ],
            ),
        ],
    )
    def test_mean_n_best(
        self,
        get_single_phase_xmap,
        nav_shape,
        rot_per_point,
        mean_n_best,
        desired_merged_scores,
        desired_merged_sim_idx,
    ):
        """Ensure that the mergesorted scores and simulation index
        properties in the merged map has the correct values and shape.
        """
        n_phases = np.shape(desired_merged_scores)[-1] // rot_per_point
        xmaps = []
        for i in range(n_phases):
            xmap = get_single_phase_xmap(nav_shape, rot_per_point, name=str(i))
            xmap[i].scores += i
            xmaps.append(xmap)

        # The simulation indices should be the same in all maps
        all_sim_idx = np.dstack([xmap.simulation_indices for xmap in xmaps])
        assert np.sum(np.diff(all_sim_idx)) == 0

        merged_xmap = merge_crystal_maps(
            crystal_maps=xmaps, mean_n_best=mean_n_best,
        )

        assert merged_xmap.phases.size == n_phases
        assert np.allclose(merged_xmap.merged_scores, desired_merged_scores)
        assert np.allclose(
            merged_xmap.merged_simulation_indices, desired_merged_sim_idx
        )

    def test_mean_n_best_varying_scores(self, get_single_phase_xmap):
        """Ensure various combinations of scores per point and how many
        of these are evaulated to find the best match return expected
        results.
        """
        nav_shape = (2, 3)
        rot_per_point = 3
        xmap1 = get_single_phase_xmap(nav_shape, rot_per_point, name="a")
        xmap2 = get_single_phase_xmap(nav_shape, rot_per_point, name="b")
        idx = (0, 0)
        xmap1[idx].scores = [1, 2, 2.1]
        xmap2[idx].scores = [1, 1.9, 3]
        xmap2[0, 1].scores = 2.0  # Both maps in both merged maps

        crystal_maps = [xmap1, xmap2]
        merged_xmap1 = merge_crystal_maps(crystal_maps, mean_n_best=2)
        merged_xmap2 = merge_crystal_maps(crystal_maps, mean_n_best=3)

        assert np.allclose(merged_xmap1.phase_id, [0, 1, 0, 0, 0, 0])
        assert np.allclose(merged_xmap2.phase_id, [1, 1, 0, 0, 0, 0])

    def test_merging_maps_different_shapes_raises(self, get_single_phase_xmap):
        xmap1 = get_single_phase_xmap((4, 3))
        xmap2 = get_single_phase_xmap((3, 4))
        with pytest.raises(ValueError, match="All crystal maps must have the"):
            _ = merge_crystal_maps([xmap1, xmap2])

    def test_merging_maps_different_number_of_scores_raises(
        self, get_single_phase_xmap
    ):
        nav_shape = (2, 3)
        xmap1 = get_single_phase_xmap(nav_shape, 3, name="a")
        xmap2 = get_single_phase_xmap(nav_shape, 4, name="b")
        xmap2[0, 1].scores = 2.0  # Both maps in both merged maps

        crystal_maps = [xmap1, xmap2]
        with pytest.raises(ValueError, match="All crystal maps must have the"):
            _ = merge_crystal_maps(crystal_maps)

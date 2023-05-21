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

import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation
import pytest

from kikuchipy.indexing import merge_crystal_maps


class TestMergeCrystalMaps:
    @pytest.mark.parametrize(
        "map_shape, rot_per_point, phase_names",
        [
            ((3,), 10, ["a", "b"]),
            ((4,), 1, ["a", "b", "c"]),
            ((3,), 5, ["austenite", "ferrite"]),
            ((4,), 1, ["al", "cu", "si"]),
        ],
    )
    def test_merge_crystal_maps_1d(
        self, map_shape, rot_per_point, phase_names, get_single_phase_xmap
    ):
        """Crystal maps with a 1D navigation shape can be merged
        successfully and yields an expected output.
        """
        n_phases = len(phase_names)
        scores_prop, sim_idx_prop = "scores", "sim_idx"

        map_size = int(np.sum(map_shape))
        data_shape = (map_size,)
        if rot_per_point > 1:
            data_shape += (rot_per_point,)

        desired_phase_ids = np.zeros(map_size)
        desired_scores = np.ones(data_shape)
        desired_idx = np.arange(np.prod(data_shape)).reshape(data_shape)

        xmaps = []
        xmap_kwargs = dict(
            nav_shape=map_shape,
            rotations_per_point=rot_per_point,
            prop_names=[scores_prop, sim_idx_prop],
            step_sizes=(1,),
        )
        phase_ids = np.arange(n_phases)
        for i in range(n_phases):
            xmap = get_single_phase_xmap(
                name=phase_names[i], phase_id=phase_ids[i], **xmap_kwargs
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
                assert np.allclose(v1, v2)

        assert np.allclose(merged_xmap.phase_id, desired_phase_ids)
        assert np.allclose(merged_xmap.prop[scores_prop], desired_scores)
        assert np.allclose(merged_xmap.prop[sim_idx_prop], desired_idx)
        assert np.allclose(merged_xmap.rotations.data, desired_rot)

        desired_merged_shapes = (map_size, rot_per_point * n_phases)
        assert merged_xmap.prop[f"merged_{scores_prop}"].shape == desired_merged_shapes
        assert merged_xmap.prop[f"merged_{sim_idx_prop}"].shape == desired_merged_shapes

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

        map_size = int(np.prod(map_shape))
        data_shape = (map_size,)
        if rot_per_point > 1:
            data_shape += (rot_per_point,)

        desired_phase_ids = np.zeros(map_size)
        desired_scores = np.ones(data_shape)
        desired_idx = np.arange(int(np.prod(data_shape))).reshape(data_shape)

        xmaps = []
        xmap_kw = dict(
            nav_shape=map_shape,
            rotations_per_point=rot_per_point,
            prop_names=[scores_prop, sim_idx_prop],
        )
        phase_ids = np.arange(n_phases)
        ny, nx = map_shape
        for i in range(n_phases):
            xmap = get_single_phase_xmap(
                name=phase_names[i], phase_id=phase_ids[i], **xmap_kw
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
            assert np.allclose(v1, v2)

        assert np.allclose(merged_xmap.phase_id, desired_phase_ids)
        assert np.allclose(merged_xmap.prop[scores_prop], desired_scores)
        assert np.allclose(merged_xmap.prop[sim_idx_prop], desired_idx)
        assert np.allclose(merged_xmap.rotations.data, desired_rot)

        desired_merged_shapes = (map_size, rot_per_point * n_phases)
        assert merged_xmap.prop[f"merged_{scores_prop}"].shape == desired_merged_shapes
        assert merged_xmap.prop[f"merged_{sim_idx_prop}"].shape == desired_merged_shapes

    @pytest.mark.parametrize(
        "scores_prop, sim_idx_prop", [("scores", "sim_idx"), ("similar", "simulated")]
    )
    def test_property_names(self, get_single_phase_xmap, scores_prop, sim_idx_prop):
        """Passing scores and simulation indices property names returns
        expected properties in merged map.
        """
        map_shape = (5, 6)
        rot_per_point = 50

        xmap1 = get_single_phase_xmap(
            map_shape, rot_per_point, [scores_prop, sim_idx_prop], "a", phase_id=0
        )
        xmap2 = get_single_phase_xmap(
            map_shape, rot_per_point, [scores_prop, sim_idx_prop], "b", phase_id=1
        )

        xmap2[3, 3].prop[scores_prop] = 2
        merged_xmap = merge_crystal_maps(
            crystal_maps=[xmap1, xmap2],
            greater_is_better=True,
            scores_prop=scores_prop,
            simulation_indices_prop=sim_idx_prop,
        )

        assert scores_prop in merged_xmap.prop.keys()
        assert sim_idx_prop in merged_xmap.prop.keys()

        desired_merged_shapes = (np.prod(map_shape), rot_per_point * 2)
        assert merged_xmap.prop[f"merged_{scores_prop}"].shape == desired_merged_shapes
        assert merged_xmap.prop[f"merged_{sim_idx_prop}"].shape == desired_merged_shapes

    def test_lower_is_better(self, get_single_phase_xmap):
        map_shape = (5, 6)
        rot_per_point = 5
        scores_prop = "scores"
        sim_idx_prop = "simulation_indices"

        xmap1 = get_single_phase_xmap(
            map_shape, rot_per_point, [scores_prop, sim_idx_prop], "a", phase_id=0
        )
        xmap2 = get_single_phase_xmap(
            map_shape, rot_per_point, [scores_prop, sim_idx_prop], "b", phase_id=1
        )

        xmap2[0, 3].prop[scores_prop] = 0
        desired_phase_id = np.zeros(np.prod(map_shape))
        desired_phase_id[3] = 1

        merged_xmap = merge_crystal_maps(
            crystal_maps=[xmap1, xmap2],
            greater_is_better=False,
            simulation_indices_prop=sim_idx_prop,
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
    def test_warning_merge_maps_with_same_phase_name(
        self, get_single_phase_xmap, phase_names, desired_phase_names
    ):
        n_phases = len(phase_names)
        scores_prop = "scores"
        sim_idx_prop = "simulated_indices"
        map_shape = (5, 6)
        rot_per_point = 5

        xmaps = []
        xmap_kw = dict(
            nav_shape=map_shape,
            rotations_per_point=rot_per_point,
            prop_names=[scores_prop, sim_idx_prop],
        )
        phase_ids = np.arange(n_phases)
        for i in range(n_phases):
            # Same name, different space groups
            xmap = get_single_phase_xmap(
                name=phase_names[i], space_group=i + 1, phase_id=phase_ids[i], **xmap_kw
            )
            xmap.phases[phase_ids[i]].space_group = i + 1
            # All maps have at least one point with the best score
            xmap[i, i].scores += i + 1
            xmaps.append(xmap)

        with pytest.warns(
            UserWarning, match=f"There are duplicates of phase '{phase_names[0]}'"
        ):
            merged_xmap = merge_crystal_maps(
                crystal_maps=xmaps,
                scores_prop=scores_prop,
                simulation_indices_prop=sim_idx_prop,
            )

        assert all([name in merged_xmap.phases.names for name in desired_phase_names])

    @pytest.mark.parametrize(
        (
            "nav_shape, rot_per_point, mean_n_best, desired_merged_scores, "
            "desired_merged_sim_idx"
        ),
        [
            ((2,), 1, 1, [[1, 1], [2, 1]], [[0, 2], [3, 1]]),
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
        """Ensure that the merge sorted scores and simulation index
        properties in the merged map has the correct values and shape.
        """
        prop_names = ["scores", "simulation_indices"]
        n_phases = np.shape(desired_merged_scores)[-1] // rot_per_point
        xmaps = []
        for i in range(n_phases):
            xmap = get_single_phase_xmap(
                nav_shape, rot_per_point, name=str(i), prop_names=prop_names
            )
            xmap[i].scores += i
            xmaps.append(xmap)

        # The simulation indices should be the same in all maps
        all_sim_idx = np.dstack([xmap.simulation_indices for xmap in xmaps])
        assert np.sum(np.diff(all_sim_idx)) == 0

        merged_xmap = merge_crystal_maps(
            crystal_maps=xmaps,
            mean_n_best=mean_n_best,
            simulation_indices_prop=prop_names[1],
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
        with pytest.raises(ValueError, match=r"Crystal maps \(and/or navigation masks"):
            _ = merge_crystal_maps([xmap1, xmap2])

    def test_merging_maps_different_number_of_scores_raises(
        self, get_single_phase_xmap
    ):
        nav_shape = (2, 3)
        xmap1 = get_single_phase_xmap(nav_shape, 3, name="a")
        xmap2 = get_single_phase_xmap(nav_shape, 4, name="b")
        xmap2[0, 1].scores = 2.0  # Both maps in both merged maps

        crystal_maps = [xmap1, xmap2]
        with pytest.raises(ValueError, match="Crystal maps must have the"):
            _ = merge_crystal_maps(crystal_maps)

    def test_merging_refined_maps(self):
        ny, nx = (3, 3)
        nav_size = ny * nx
        rot = Rotation.from_euler(np.ones((nav_size, 3)))
        x = np.tile(np.arange(ny), nx)
        y = np.repeat(np.arange(nx), ny)

        # Simulation indices
        n_sim_indices = 10
        sim_indices1 = np.random.randint(
            low=0, high=1000, size=n_sim_indices * nav_size
        ).reshape((nav_size, n_sim_indices))
        sim_indices2 = np.random.randint(
            low=0, high=1000, size=n_sim_indices * nav_size
        ).reshape((nav_size, n_sim_indices))

        # Scores
        scores1 = np.ones(nav_size)
        scores1[0] = 3
        scores2 = 2 * np.ones(nav_size)

        xmap1 = CrystalMap(
            rotations=rot,
            phase_id=np.ones(nav_size) * 0,
            phase_list=PhaseList(Phase(name="a")),
            x=x,
            y=y,
            prop={"simulation_indices": sim_indices1, "scores": scores1},
        )
        xmap2 = CrystalMap(
            rotations=rot,
            phase_id=np.ones(nav_size),
            phase_list=PhaseList(Phase(name="b")),
            x=x,
            y=y,
            prop={"simulation_indices": sim_indices2, "scores": scores2},
        )
        xmap_merged = merge_crystal_maps(crystal_maps=[xmap1, xmap2])

        assert "simulation_indices" not in xmap_merged.prop.keys()
        assert "merged_simulation_indices" not in xmap_merged.prop.keys()

        with pytest.raises(ValueError, match="Cannot merge maps with more"):
            _ = merge_crystal_maps(
                crystal_maps=[xmap1, xmap2],
                simulation_indices_prop="simulation_indices",
            )

    def test_merging_with_navigation_masks(self):
        # Setup map 1
        xmap1 = CrystalMap.empty((3, 4))
        xmap1.prop["scores"] = np.arange(xmap1.size)
        xmap1.prop["simulation_indices"] = np.arange(xmap1.size)
        xmap1.phases[0].name = "a"
        # Setup map 2
        xmap2 = xmap1.deepcopy()
        xmap2.phase_id = 1
        xmap2.phases = PhaseList(names="b", ids=1)
        xmap2.simulation_indices += xmap2.size
        xmap2[0, 0].scores = 1

        # Works without masks
        xmap3 = merge_crystal_maps(
            [xmap1, xmap2], simulation_indices_prop="simulation_indices"
        )
        # fmt: off
        assert np.allclose(
            xmap3.phase_id,
            [
                1, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
            ]
        )
        assert np.allclose(
            xmap3.scores,
            [
                1, 1,  2,  3,
                4, 5,  6,  7,
                8, 9, 10, 11,
            ]
        )
        assert np.allclose(
            xmap3.simulation_indices,
            [
                12,  1,  2,  3,
                 4,  5,  6,  7,
                 8,  9, 10, 11,
            ]
        )
        # fmt: on

        # Use internal masks via CrystalMap.is_in_data (xmap[:1, 1:] is
        # a view and not a copy)
        xmap4 = merge_crystal_maps(
            [xmap1[1:, 1:], xmap2[1:, 1:]], simulation_indices_prop="simulation_indices"
        )
        # fmt: off
        assert np.allclose(
            xmap4.phase_id,
            [
                0, 0, 0,
                0, 0, 0,
            ]
        )
        assert np.allclose(
            xmap4.scores,
            [
                5,  6,  7,
                9, 10, 11,
            ]
        )
        assert np.allclose(
            xmap4.simulation_indices,
            [
                5,  6,  7,
                9, 10, 11,
            ]
        )
        # fmt: on

        nav_mask1 = np.ones(xmap1.shape, dtype=bool)
        nav_mask1[1:, 1:] = False
        nav_mask2 = ~nav_mask1

        # Equal number of a map's points in data and number of False
        # entries in a navigation mask
        xmap5 = merge_crystal_maps(
            [xmap1[~nav_mask1.ravel()], xmap2[~nav_mask2.ravel()]],
            navigation_masks=[nav_mask1, nav_mask2],
            simulation_indices_prop="simulation_indices",
        )
        # fmt: off
        assert np.allclose(
            xmap5.phase_id,
            [
                1, 1, 1, 1,
                1, 0, 0, 0,
                1, 0, 0, 0
            ]
        )
        assert np.allclose(
            xmap5.simulation_indices,
            [
                12, 13, 14, 15,
                16,  5,  6,  7,
                20,  9, 10, 11,
            ]
        )
        # fmt: on

        # All points in one map should be used, but not in another:
        # Only consider xmap1 in the first row and first column (mask it
        # out everywhere else)
        xmap6 = merge_crystal_maps(
            [xmap1[nav_mask1.ravel()], xmap2], navigation_masks=[~nav_mask1, None]
        )
        # fmt: off
        assert np.allclose(
            xmap6.phase_id,
            [
                1, 0, 0, 0,
                0, 1, 1, 1,
                0, 1, 1, 1,
            ]
        )
        # fmt: on

        # Only consider xmap1 in the lower right corner (mask it out
        # everywhere else)
        xmap7 = merge_crystal_maps(
            [xmap1[~nav_mask1.ravel()], xmap2], navigation_masks=[nav_mask1, None]
        )
        # fmt: off
        assert np.allclose(
            xmap7.phase_id,
            [
                1, 1, 1, 1,
                1, 0, 0, 0,
                1, 0, 0, 0,
            ]
        )
        # fmt: on

    def test_merging_with_navigation_masks_equal_phase(self):
        # Setup map 1
        xmap1 = CrystalMap.empty((3, 4))
        xmap1.prop["scores"] = np.arange(xmap1.size)
        xmap1.phases[0].name = "a"
        # Setup map 2
        xmap2 = xmap1.deepcopy()
        xmap2.phase_id = 1
        xmap2.phases = PhaseList(names="a", ids=1)
        xmap2[0, 0].scores = 1

        xmap3 = merge_crystal_maps([xmap1, xmap2])
        assert np.allclose(xmap3.phase_id, 0)

    def test_merging_with_navigation_masks_raises(self):
        # Setup map 1
        xmap1 = CrystalMap.empty((3, 4))
        xmap1.prop["scores"] = np.arange(xmap1.size)
        xmap1.phases[0].name = "a"
        # Setup map 2
        xmap2 = xmap1.deepcopy()
        xmap2.phase_id = 1
        xmap2.phases = PhaseList(names="b", ids=1)
        xmap2[0, 0].scores = 1

        nav_mask1 = np.ones(xmap1.shape, dtype=bool)
        nav_mask1[1:, 1:] = False
        nav_mask2 = ~nav_mask1

        # Unequal number of maps and masks
        with pytest.raises(ValueError, match="Number of crystal maps and navigation "):
            _ = merge_crystal_maps([xmap1, xmap2], navigation_masks=nav_mask1)

        # Unequal shape of maps
        with pytest.raises(ValueError, match=r"Crystal maps \(and/or navigation masks"):
            _ = merge_crystal_maps(
                [xmap1[~nav_mask1.ravel()], xmap2[~nav_mask2.ravel()]],
            )

        # Must be as many points in a map's data as there are False
        # entries in a mask
        with pytest.raises(ValueError, match="0. navigation mask does not have as "):
            _ = merge_crystal_maps(
                [xmap1, xmap2], navigation_masks=[nav_mask1, nav_mask2]
            )

        # A mask is not a NumPy array
        with pytest.raises(ValueError, match="1. navigation mask must be a NumPy "):
            _ = merge_crystal_maps(
                [xmap1[~nav_mask1.ravel()], xmap2[~nav_mask2.ravel()]],
                navigation_masks=[nav_mask1, list(nav_mask2)],
            )

    def test_not_indexed(self):
        xmap_a = CrystalMap.empty((4, 3))
        is_indexed_a = np.array(
            [[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1]], dtype=bool
        ).ravel()
        xmap_a.phases.add_not_indexed()
        xmap_a.phases[0].name = "a"
        xmap_a[~is_indexed_a].phase_id = -1
        xmap_a.prop["scores"] = np.array(
            [[2, 2, 0], [3, 0, 4], [0, 4, 3], [0, 2, 1]], dtype=float
        ).ravel()
        xmap_a._rotations = xmap_a.rotations * Rotation.from_axes_angles(
            [0, 0, 1], 30, degrees=True
        )

        xmap_b = CrystalMap.empty((4, 3))
        is_indexed_b = np.array(
            [[1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0]], dtype=bool
        ).ravel()
        xmap_b.phases.add_not_indexed()
        xmap_b.phases[0].name = "b"
        xmap_b[~is_indexed_b].phase_id = -1
        xmap_b.prop["scores"] = np.array(
            [[3, 1, 0], [2, 1, 5], [0, 2, 4], [0, 1, 0]], dtype=float
        ).ravel()
        xmap_b._rotations = xmap_b.rotations * Rotation.from_axes_angles(
            [0, 0, 1], 60, degrees=True
        )

        xmap_ab = merge_crystal_maps([xmap_a, xmap_b])

        assert np.allclose(xmap_ab.phase_id, [1, 0, -1, 0, 1, 1, -1, 0, 1, -1, 0, 0])
        assert np.allclose(
            xmap_ab["indexed"].rotations.angle,
            np.deg2rad([60, 30, 30, 60, 60, 30, 60, 30, 30]),
        )
        assert np.allclose(xmap_ab["indexed"].scores, [3, 2, 3, 1, 5, 4, 4, 2, 1])

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

from orix.crystal_map import CrystalMap, PhaseList, Phase
from orix.quaternion import Rotation
import numpy as np
import pytest

from kikuchipy.indexing._merge_crystal_maps import merge_crystal_maps


class TestMergeCrystalMaps:
    def test_merge_crystal_maps_1d(self):
        n_rot_per_point = 2
        nav_size = 9
        x = np.arange(nav_size)

        prop_shape = (nav_size, n_rot_per_point)
        scores = np.ones(prop_shape)
        sim_idx1 = np.arange(np.prod(prop_shape)).reshape(prop_shape)
        sim_idx2 = np.arange(np.prod(prop_shape)).reshape(prop_shape)

        score_prop = "scores"
        sim_idx_prop = "simulated_indices"

        r = Rotation(np.zeros((nav_size, 4)))
        xmap1 = CrystalMap(
            rotations=r,
            x=x,
            phase_list=PhaseList(Phase("a")),
            prop={score_prop: scores, sim_idx_prop: sim_idx1},
        )
        xmap2 = CrystalMap(
            rotations=r,
            x=x,
            phase_list=PhaseList(Phase("b")),
            prop={score_prop: scores, sim_idx_prop: sim_idx2},
        )

        merged_xmap = merge_crystal_maps(
            crystal_maps=[xmap2, xmap1],
            score_prop=score_prop,
            simulation_indices_prop=sim_idx_prop,
        )

        assert np.allclose(
            merged_xmap.prop[sim_idx_prop][:, 0],
            np.linspace(0, np.prod(prop_shape) - n_rot_per_point, nav_size),
        )

    def test_merge_crystal_maps_2d(self):
        pass

    def test_single_scores_per_point(self):
        pass

    def test_multiple_scores_per_point(self):
        pass

    def test_more_indices_than_scores_per_point(self):
        pass

    def test_more_scores_than_indices_per_point(self):
        pass

    def test_merging_maps_different_shapes_raises(self):
        pass

    def test_merging_maps_different_number_of_scores(self):
        pass

    @pytest.mark.parametrize(
        "crystal_map_input, score_prop, sim_idx_prop",
        [
            (((4, 3), (1, 1), 10, [0]), "scores", "sim_idx"),
            (((3, 4), (1.5, 1.5), 5, [0]), "similar", "simulated"),
        ],
        indirect=["crystal_map_input"],
    )
    def test_property_names(self, crystal_map_input, score_prop, sim_idx_prop):
        """Passing scores and simulation indices property names returns
        expected properties in merged map.
        """
        crystal_map_input["phase_list"] = PhaseList(Phase("a"))
        xmap1 = CrystalMap(**crystal_map_input)
        m = xmap1.size
        n = xmap1.rotations_per_point
        mn = m * n

        crystal_map_input["phase_list"] = PhaseList(Phase("b"))
        xmap2 = CrystalMap(**crystal_map_input)

        xmap1.prop[score_prop] = np.random.random(mn).reshape((m, n))
        xmap2.prop[score_prop] = np.random.random(mn).reshape((m, n))

        sim_idx = np.arange(m)
        xmap1.prop[sim_idx_prop] = np.random.choice(sim_idx, mn).reshape((m, n))
        xmap2.prop[sim_idx_prop] = np.random.choice(sim_idx, mn).reshape((m, n))

        merged_xmap = merge_crystal_maps(
            crystal_maps=[xmap1, xmap2],
            mean_n_best=n,
            score_prop=score_prop,
            simulation_indices_prop=sim_idx_prop,
        )

        assert score_prop in merged_xmap.prop.keys()
        assert sim_idx_prop in merged_xmap.prop.keys()

    def test_negative_metric(self):
        pass

    def test_merging_n_maps(self):
        pass

    def test_merging_returns_same_map(self):
        pass

    def test_warning_merge_maps_with_same_phase(self):
        score_prop = "scores"
        sim_idx_prop = "simulated_indices"
        nav_size = 9

        r = Rotation(np.ones((nav_size, 4)))
        scores1 = np.ones(nav_size)
        idx = nav_size - 2
        scores1[idx] = 0

        xmap1 = CrystalMap(
            rotations=r,
            phase_list=PhaseList(Phase("a")),
            prop={score_prop: scores1, sim_idx_prop: np.arange(nav_size)},
        )
        scores2 = np.ones(nav_size)
        scores2[idx] += 1
        xmap2 = CrystalMap(
            rotations=r,
            phase_list=PhaseList(Phase("a")),
            prop={score_prop: scores2, sim_idx_prop: np.arange(nav_size) * 2},
        )

        with pytest.warns(UserWarning):
            merged_xmap = merge_crystal_maps(
                crystal_maps=[xmap1, xmap2],
                score_prop=score_prop,
                simulation_indices_prop=sim_idx_prop,
            )

        desired_phase_id = np.zeros(nav_size)
        desired_phase_id[idx] = 1
        assert np.allclose(merged_xmap.phase_id, desired_phase_id)

    def test_combined_scores_shape(self):
        pass

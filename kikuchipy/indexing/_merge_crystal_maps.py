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

from math import copysign
from typing import List, Optional, Union
import warnings

import numpy as np
from orix.crystal_map import create_coordinate_arrays, CrystalMap, PhaseList
from orix.quaternion import Rotation

from kikuchipy.signals.util._crystal_map import _equal_phase


def merge_crystal_maps(
    crystal_maps: List[CrystalMap],
    mean_n_best: int = 1,
    greater_is_better: Optional[int] = None,
    scores_prop: str = "scores",
    simulation_indices_prop: Optional[str] = None,
    navigation_masks: Optional[List[Union[None, np.ndarray]]] = None,
) -> CrystalMap:
    """Return a multi phase :class:`~orix.crystal_map.CrystalMap` by
    merging maps of 1D or 2D navigation shape based on scores.

    It is required that all maps have the same number of rotations and
    scores (and simulation indices if applicable) per point.

    Parameters
    ----------
    crystal_maps
        A list of at least two crystal maps with simulated indices and
        scores among their properties. The maps must have the same
        shape, unless navigation masks are passed (see
        ``navigation_masks``). Identical phases are considered as one
        phase in the returned map.
    mean_n_best
        Number of best metric results to take the mean of before
        comparing. Default is ``1``. If given with a negative sign and
        ``greater_is_better`` is not given, the n lowest valued metric
        results are chosen.
    greater_is_better
        ``True`` if a higher score means a better match. If not given,
        the sign of ``mean_n_best`` is used, with a positive sign
        meaning ``True``.
    scores_prop
        Name of scores array in the crystal maps' properties. Default
        is ``"scores"``.
    simulation_indices_prop
        Name of simulated indices array in the crystal maps' properties.
        If not given (default), the merged crystal map will not contain
        an array of merged simulation indices from the input crystal
        maps' properties. If a string, there must be as many simulation
        indices per point as there are scores.
    navigation_masks
        A list of boolean masks of shapes equal to the full 1D or 2D
        navigation (map) shape, where only points equal to ``False`` are
        considered when comparing scores. The number of ``False``
        entries in a mask must be equal to the number of points in a
        crystal map (:attr:`~orix.crystal_map.CrystalMap.size`). The
        order corresponds to the order in ``crystal_maps``. If not
        given, all points are used. If all points in one or more of the
        maps should be used, this map's entry can be ``None``.

    Returns
    -------
    merged_xmap
        A crystal map where the rotation of the phase with the best
        matching score(s) is assigned to each point. The best matching
        scores, merge sorted, are added to its properties with a name
        equal to whatever passed to ``scores_prop`` with "merged" as a
        suffix. If ``simulation_indices_prop`` is passed, the best
        matching simulation indices are added in the same way as the
        scores.

    Notes
    -----
    The initial motivation behind this function was to merge single
    phase maps produced by dictionary indexing.
    """
    n_maps = len(crystal_maps)

    # Set `navigation_masks` if any of the maps have some points not in
    # the data
    if navigation_masks is None:
        all_is_in_data = [xmap.is_in_data.all() for xmap in crystal_maps]
        if not all(all_is_in_data):
            navigation_masks = []
            for xmap in crystal_maps:
                slice_i = xmap._data_slices_from_coordinates()
                is_in_data2d_i = xmap.is_in_data.reshape(xmap._original_shape)[slice_i]
                navigation_masks.append(~is_in_data2d_i)

    # Get map shapes of all maps. We can get this from either
    # `CrystalMap.shape` or `mask.shape` for masks in `navigation_masks`
    if navigation_masks is not None:
        if len(navigation_masks) != n_maps:
            raise ValueError(
                "Number of crystal maps and navigation masks must be equal"
            )

        map_shapes = []
        for i, (mask, xmap) in enumerate(zip(navigation_masks, crystal_maps)):
            if isinstance(mask, np.ndarray):
                mask_is_in_data = np.sum(~mask)
                map_is_in_data = xmap.is_in_data.sum()
                if mask_is_in_data != map_is_in_data:
                    raise ValueError(
                        f"{i}. navigation mask does not have as many 'False', "
                        f"{mask_is_in_data}, as there are points in the crystal map, "
                        f"{map_is_in_data}"
                    )
                map_shapes.append(mask.shape)
            elif mask is None:
                map_shapes.append(xmap.shape)
            else:
                raise ValueError(
                    f"{i}. navigation mask must be a NumPy array or 'None'"
                )
    else:
        map_shapes = [xmap.shape for xmap in crystal_maps]

    if not np.sum(abs(np.diff(map_shapes, axis=0))) == 0:
        raise ValueError(
            "Crystal maps (and/or navigation masks) must have the same navigation shape"
        )
    else:
        map_shape = map_shapes[0]
        map_size = int(np.prod(map_shape))

    if navigation_masks is not None:
        navigation_masks1d = []
        for mask, map_shape in zip(navigation_masks, map_shapes):
            map_size = int(np.prod(map_shape))
            if mask is None:
                mask1d = np.ones(map_size, dtype=bool)
            else:
                mask1d = ~mask.ravel()
            navigation_masks1d.append(mask1d)
    else:
        navigation_masks1d = [None] * n_maps

    rot_per_point_per_map = [xmap.rotations_per_point for xmap in crystal_maps]
    if not all(np.diff(rot_per_point_per_map) == 0):
        raise ValueError(
            "Crystal maps must have the same number of rotations and scores per point"
        )
    else:
        n_scores_per_point = rot_per_point_per_map[0]

    if simulation_indices_prop is not None:
        n_sim_idx = crystal_maps[0].prop[simulation_indices_prop].shape
        if len(n_sim_idx) > 1 and n_sim_idx[1] > n_scores_per_point:
            raise ValueError(
                "Cannot merge maps with more simulation indices than scores per point"
            )

    if greater_is_better is None:
        sign = copysign(1, mean_n_best)
        mean_n_best = abs(mean_n_best)
    else:
        if greater_is_better:
            sign = 1
        else:
            sign = -1

    # Notation used in the comments below:
    # - M: number of map points
    # - N: number of scores per point
    # - I: number of simulation indices per point
    # - K: number of maps to merge

    # Shape of the combined (unsorted) scores array, and the total
    # number of scores per point. Shape: (M, N, K) or (M, K) if only one
    # score is available (e.g. refined dot products from EMsoft)
    comb_shape = (map_size,)
    if n_scores_per_point > 1:
        comb_shape += (n_scores_per_point,)
    comb_shape += (n_maps,)

    # Combined (unsorted) scores array of shape (M, N, K) or (M, K)
    scores_dtype = crystal_maps[0].prop[scores_prop].dtype
    combined_scores = np.full(
        comb_shape, np.nan, dtype=np.dtype(f"f{scores_dtype.itemsize}")
    )
    for i, (mask, xmap) in enumerate(zip(navigation_masks1d, crystal_maps)):
        if mask is not None:
            combined_scores[mask, ..., i] = xmap.prop[scores_prop]
        else:
            combined_scores[..., i] = xmap.prop[scores_prop]

    # Best score in each map point
    if n_scores_per_point > 1:  # (M, N, K) -> (M, K)
        best_scores = combined_scores[:, :mean_n_best].squeeze()
        if len(best_scores.shape) > 2:
            best_scores = np.nanmean(best_scores, axis=1)
    else:  # (M, K)
        best_scores = combined_scores

    # Phase of best score in each map point
    phase_id = np.nanargmax(sign * best_scores, axis=1)

    # Set the phase ID of points marked as not-indexed in all maps to -1
    not_indexed = np.zeros((n_maps, map_size), dtype=bool)
    for i in range(n_maps):
        mask = navigation_masks1d[i]
        xmap = crystal_maps[i]
        if mask is not None:
            not_indexed[i, mask][xmap.phase_id == -1] = True
        else:
            not_indexed[i, xmap.phase_id == -1] = True
    not_indexed = np.logical_and.reduce(not_indexed)
    phase_id[not_indexed] = -1

    # Get the new crystal map's rotations, scores and indices,
    # restricted to one phase per point (uncombined)
    new_rotations = np.zeros(comb_shape[:-1] + (4,), dtype="float")
    new_scores = np.zeros(comb_shape[:-1], dtype=scores_dtype)

    if simulation_indices_prop is not None:
        new_indices = np.zeros(comb_shape[:-1], dtype="int32")

    phase_list = PhaseList()
    if -1 in phase_id:
        phase_list.add_not_indexed()
    for i, (nav_mask1d, xmap) in enumerate(zip(navigation_masks1d, crystal_maps)):
        phase_mask = phase_id == i

        if phase_mask.any():
            phase_ids = xmap.phases_in_data.ids
            if -1 in phase_ids:
                phase_ids.remove(-1)
            phase = xmap.phases_in_data[phase_ids[0]].deepcopy()
            if phase.name in phase_list.names:
                # If they are equal, do not duplicate it in the phase
                # list but update the phase ID
                equal_phases, different = _equal_phase(phase, phase_list[phase.name])
                if equal_phases:
                    phase_id[phase_mask] = phase_list.id_from_name(phase.name)
                else:
                    name = phase.name
                    phase.name = name + str(i)
                    warnings.warn(
                        f"There are duplicates of phase '{name}' but the phases have "
                        f"different {different}, will therefore rename this phase's "
                        f"name to '{phase.name}' in the merged PhaseList",
                    )
                    phase_list.add(phase)
            else:
                phase_list.add(phase)
        else:
            continue

        if nav_mask1d is not None:
            phase_mask2 = phase_mask[nav_mask1d]
            new_rotations[phase_mask] = xmap.rotations[phase_mask2].data
            new_scores[phase_mask] = xmap.prop[scores_prop][phase_mask2]
        else:
            # Old behavior
            new_rotations[phase_mask] = xmap.rotations[phase_mask].data
            new_scores[phase_mask] = xmap.prop[scores_prop][phase_mask]

        if simulation_indices_prop is not None:
            if nav_mask1d is not None:
                new_indices[phase_mask] = xmap.prop[simulation_indices_prop][
                    phase_mask2
                ]
            else:
                # Old behavior
                new_indices[phase_mask] = xmap.prop[simulation_indices_prop][phase_mask]

    # To get the combined, best, sorted scores and simulation indices
    # from all maps (phases), we collapse the second and (potentially)
    # third axis to get (M, N * K) or (M, K)
    mergesort_shape = (comb_shape[0], np.prod(comb_shape[1:]))
    comb_scores_reshaped = combined_scores.reshape(mergesort_shape)
    best_sorted_idx = np.argsort(sign * -comb_scores_reshaped, kind="mergesort", axis=1)

    # Best, sorted scores in all maps (for all phases) per point
    merged_best_scores = np.take_along_axis(
        comb_scores_reshaped, best_sorted_idx, axis=-1
    )

    # Set up merged map's properties
    props = {scores_prop: new_scores, f"merged_{scores_prop}": merged_best_scores}

    if simulation_indices_prop is not None:
        # Combined (unsorted) simulation indices array of shape
        # (M, N, K) or (M, K), accounting for the case where there are
        # more simulation indices per point than scores (e.g. refined
        # dot products from EMsoft)
        comb_sim_idx_list = []
        for i, (nav_mask1d, xmap) in enumerate(zip(navigation_masks1d, crystal_maps)):
            if nav_mask1d is not None:
                sim_idx_i = np.full(comb_shape[:-1], np.nan)
                sim_idx_i[nav_mask1d] = xmap.prop[simulation_indices_prop]
            else:
                sim_idx_i = xmap.prop[simulation_indices_prop]
            comb_sim_idx_list.append(sim_idx_i)

        comb_sim_idx = np.dstack(comb_sim_idx_list)

        # To enable calculation of an orientation similarity map from
        # the combined, sorted simulation indices array, we must make
        # the indices unique across all maps
        for i in range(1, comb_sim_idx.shape[-1]):
            increment = (
                abs(
                    np.nanmax(comb_sim_idx[..., i - 1])
                    - np.nanmin(comb_sim_idx[..., i])
                )
                + 1
            )
            comb_sim_idx[..., i] += increment

        # Collapse axes as for the combined scores array above
        comb_sim_idx = comb_sim_idx.reshape(mergesort_shape)

        # Best, sorted simulation indices in all maps (for all phases)
        # per point
        merged_simulated_indices = np.take_along_axis(
            comb_sim_idx, best_sorted_idx, axis=-1
        )

        # Finally, add to properties
        props[simulation_indices_prop] = new_indices
        props[f"merged_{simulation_indices_prop}"] = merged_simulated_indices

    step_sizes = (crystal_maps[0].dx, crystal_maps[0].dy)
    coords, _ = create_coordinate_arrays(
        map_shape, step_sizes=step_sizes[: len(map_shape)]
    )

    return CrystalMap(
        rotations=Rotation(new_rotations),
        phase_id=phase_id,
        phase_list=phase_list,
        prop=props,
        scan_unit=crystal_maps[0].scan_unit,
        **coords,
    )

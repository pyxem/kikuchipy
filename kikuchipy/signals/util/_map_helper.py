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

from typing import Callable, Union

import numpy as np
from scipy.ndimage import generic_filter

from kikuchipy.indexing.similarity_metrics import (
    _normalize_expt_sim,
    _zero_mean_expt_sim,
)
from kikuchipy.filters import Window


def _map_helper(
    patterns: np.ndarray,
    map_function: Callable,
    window: Window,
    nav_shape: tuple,
    dtype_out: type = np.float32,
    ignore_map: bool = False,
    **kwargs,
) -> np.ndarray:
    # Array in the original navigation shape of indices into the
    # flattened navigation axis
    flat_index_map = np.arange(np.prod(nav_shape)).reshape(nav_shape)

    def wrapped_map_function(indices):
        # Indices to be picked out with window from flat_index_map
        indices = indices.astype(int)
        return map_function(
            patterns=patterns,
            indices=indices,
            nav_shape=nav_shape,
            window=window,
            dtype_out=dtype_out,
            **kwargs,
        )

    #    dtype_out = np.bool_ if ignore_map else dtype_out
    dtype_out = bool if ignore_map else dtype_out

    return generic_filter(
        input=flat_index_map,
        function=wrapped_map_function,
        footprint=window,
        mode="constant",
        cval=-1,
        output=dtype_out,
    )


def _neighbour_dot_products(
    patterns: np.ndarray,
    indices: np.ndarray,
    nav_shape: tuple,
    sig_size: tuple,
    window: Window,
    dtype_out: type,
    output: np.ndarray,
    center_index: int,
    flat_window_truthy_indices: np.ndarray,
    standardize: bool,
    return_average: bool,
) -> Union[np.ndarray, int]:
    # Flat navigation index corresponding to the origin of the window,
    # i.e. into `patterns`, i.e. the current navigation point to average
    current_nav_idx = indices[center_index]

    # Indices into `indices` of neighbours to compute dot product with,
    # excluding neighbours outside the map and itself
    neighbour_idx = np.where((indices != current_nav_idx) & (indices != -1))[0]
    neighbours = indices[neighbour_idx]
    neighbours = np.unravel_index(neighbours, nav_shape)

    # Flat array of neighbour patterns
    neighbour_patterns = (
        patterns[neighbours].reshape((-1, sig_size)).astype(dtype_out)
    )

    # Flat pattern corresponding to the window origin, i.e. the current
    # navigation point to average
    pattern = patterns[np.unravel_index(current_nav_idx, nav_shape)]
    pattern = pattern.squeeze().flatten().astype(dtype_out)

    if standardize:
        pattern, neighbour_patterns = _zero_mean_expt_sim(
            pattern, neighbour_patterns, flat=True
        )
        pattern, neighbour_patterns = _normalize_expt_sim(
            pattern, neighbour_patterns, flat=True
        )

    dot_products = (neighbour_patterns @ pattern.T).squeeze()  # np.einsum

    # Returns average with neighbours used by _get_average_dot_product_map
    if return_average:
        return np.mean(dot_products)

    # output variable is modified in place
    if standardize:
        # Set center of similarity matrix 1.0
        output[current_nav_idx][
            flat_window_truthy_indices[center_index]
        ] = output[current_nav_idx][
            flat_window_truthy_indices[center_index]
        ] = 1.0
    else:
        # Compute dot product with itself
        # Should be the maximum value in the matrix
        output[current_nav_idx][flat_window_truthy_indices[center_index]] = (
            pattern ** 2
        ).sum()

    output[current_nav_idx][
        flat_window_truthy_indices[where_neighbours_in_indices]
    ] = dot_products

    # output variable is modified in place
    # but need to return a value
    return 0


def _get_neighbour_dot_product_matrices(
    patterns: np.ndarray,
    window: Window,
    standardize: bool = False,
    dtype_out: type = np.float32,
) -> np.ndarray:
    flat_window_origin = np.ravel_multi_index(window.origin, window.shape)
    boolean_window = window.copy().astype(bool)
    flat_window = boolean_window.flatten()
    flat_window_truthy_indices = np.nonzero(flat_window)[0]
    zeros_before_origin = np.count_nonzero(~flat_window[:flat_window_origin])
    center_index = flat_window_origin - zeros_before_origin
    nav_shape = patterns.shape[:-2]
    output = np.empty((np.prod(nav_shape), window.size), dtype_out)
    output[:] = np.nan
    _map_helper(
        patterns,
        _neighbour_dot_products,
        window=boolean_window,
        nav_shape=nav_shape,
        dtype_out=dtype_out,
        ignore_map=True,
        output=output,
        center_index=center_index,
        flat_window_truthy_indices=flat_window_truthy_indices,
        standardize=standardize,
        return_average=False,
    )
    output = output.reshape(*nav_shape, *window.shape)
    return output


def _get_average_dot_product_map(
    patterns: np.ndarray,
    window: Window,
    dtype_out: type,
    standardize: bool,
    nav_shape: tuple,
    sig_size: tuple,
) -> np.ndarray:
    """Get the average dot product map for a chunk of patterns.
    
    Parameters
    ----------
    patterns
        Pattern chunk.
    window
        Window defining the neighbours to calculate the average with.
    dtype_out
        Data type of output map.
    standardize
        Whether to standardize pattern intensities before computing the
        average dot product.
    nav_shape
        Navigation/map shape.
    sig_size
        Number of pattern pixels.
    
    Returns
    -------
    adp
        Average dot product map for the chunk of patterns.
    """
    # Index of window origin in flattened window
    flat_window_origin = np.ravel_multi_index(window.origin, window.shape)

    # Make window flat with boolean values
    boolean_window = window.copy().astype(bool)
    flat_window = boolean_window.flatten()

    # Index of window origin in boolean array with only True values
    flat_window_truthy_indices = np.nonzero(flat_window)[0]
    center_index = np.where(flat_window_truthy_indices == flat_window_origin)[
        0
    ][0]

    adp = _map_helper(
        patterns=patterns,
        map_function=_neighbour_dot_products,
        window=boolean_window,
        nav_shape=nav_shape,
        sig_size=sig_size,
        dtype_out=dtype_out,
        ignore_map=False,
        output=None,
        center_index=center_index,
        flat_window_truthy_indices=flat_window_truthy_indices,
        standardize=standardize,
        return_average=True,
    )

    return adp

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

from typing import Tuple, List, Callable, Union

import numpy as np
from scipy.ndimage import generic_filter
import dask.array as da

from kikuchipy.indexing.similarity_metrics import (
    _zero_mean,
    _normalize,
)
from kikuchipy.filters import Window


def _map_helper(
    patterns: np.ndarray,
    map_function: Callable,
    window: Window,
    nav_shape,
    dtype: np.dtype = np.float32,
    ignore_map: bool = False,
    **kwargs,
):
    # Check if Window is binary, raise error or binarize+warn if not

    flat_index_map = np.arange(np.prod(nav_shape)).reshape(nav_shape)

    def wrapped_map_function(indices):
        # indices picked out with window from flat_index_map
        indices = indices.astype(np.int)
        return map_function(
            patterns, indices, nav_shape, window, dtype, **kwargs
        )

    output = np.bool_ if ignore_map else dtype
    return generic_filter(
        flat_index_map,
        wrapped_map_function,
        footprint=window,
        mode="constant",
        cval=-1,
        output=output,
    )


def _assert_window_is_binary(window: Window):
    return None


def _neighbour_dot_products(
    patterns: np.ndarray,
    indices: np.ndarray,
    nav_shape: Tuple,
    window: Window,
    dtype: np.dtype,
    output: np.ndarray,
    center_index: int,
    flat_window_truthy_indices: np.ndarray,
    standardize: bool,
):
    # Flat navigation index corresponding with origin of window
    px = indices[center_index]

    # Neighbours navigation index
    where_neighbours_in_indices = np.where((indices != -1) & (indices != px))
    neighbours = indices[where_neighbours_in_indices]
    neighbours = np.unravel_index(neighbours, nav_shape)

    sig_size = np.prod(patterns.shape[-2:])

    # Neighbouring flat patterns
    neighbour_patterns = (
        patterns[neighbours].reshape((-1, sig_size)).astype(dtype)
    )

    # Flat pattern corresponding with origin
    pattern = (
        patterns[np.unravel_index(px, nav_shape)]
        .squeeze()
        .flatten()
        .astype(dtype)
    )

    if standardize:
        # TODO: create function for this in this file instead
        pattern, neighbour_patterns = _zero_mean(
            pattern, neighbour_patterns, flat=True
        )
        pattern, neighbour_patterns = _normalize(
            pattern, neighbour_patterns, flat=True
        )
        # Set center of similarity matrix 1.0
        output[px][flat_window_truthy_indices[center_index]] = output[px][
            flat_window_truthy_indices[center_index]
        ] = 1.0
    else:
        # Compute dot product with itself
        # Should be the maximum value in the matrix
        output[px][flat_window_truthy_indices[center_index]] = (
            pattern ** 2
        ).sum()

    dot_products = (neighbour_patterns @ pattern.T).squeeze()  # np.einsum
    output[px][
        flat_window_truthy_indices[where_neighbours_in_indices]
    ] = dot_products
    return np.mean(dot_products)


def _get_neighbour_dot_product_matrices(
    patterns: np.ndarray,
    window: Window,
    standardize: bool = False,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    _assert_window_is_binary(window)
    flat_window_origin = np.ravel_multi_index(window.origin, window.shape)
    boolean_window = window.copy().astype(bool)
    flat_window = boolean_window.flatten()
    flat_window_truthy_indices = np.nonzero(flat_window)[0]
    zeros_before_origin = np.count_nonzero(~flat_window[:flat_window_origin])
    center_index = flat_window_origin - zeros_before_origin
    nav_shape = patterns.shape[:-2]
    output = np.empty((np.prod(nav_shape), window.size), dtype)
    output[:] = np.nan
    _map_helper(
        patterns,
        _neighbour_dot_products,
        window=boolean_window,
        nav_shape=nav_shape,
        dtype=dtype,
        ignore_map=True,
        output=output,
        center_index=center_index,
        flat_window_truthy_indices=flat_window_truthy_indices,
        standardize=standardize,
    )
    output = output.reshape(*nav_shape, *window.shape)
    return output


def _adp(
    patterns: np.ndarray,
    indices: np.ndarray,
    nav_shape: Tuple,
    window: Window,
    dtype: np.dtype,
    center_index: int,
    standardize: bool,
):
    px = indices[center_index]
    neighbours = indices[np.where((indices != -1) & (indices != px))]
    sig_size = np.prod(patterns.shape[-2:])
    neighbours = np.unravel_index(neighbours, nav_shape)
    n_data = patterns[neighbours].reshape((-1, sig_size)).astype(dtype)
    p = (
        patterns[np.unravel_index(px, nav_shape)]
        .squeeze()
        .flatten()
        .astype(dtype)
    )
    if standardize:
        p, n_data = _zero_mean(p, n_data, flat=True)
        p, n_data = _normalize(p, n_data, flat=True)
    return np.mean(n_data @ p.T)


def _get_average_dot_product_map(
    patterns: np.ndarray,
    window: np.ndarray = None,
    dtype=np.float32,
    standardize: bool = False,
):
    nav_shape = patterns.shape[:-2]
    flat_window_origin = np.ravel_multi_index(window.origin, window.shape)
    boolean_window = window.copy().astype(bool)
    flat_window = boolean_window.flatten()
    zeros_before_origin = np.count_nonzero(~flat_window[:flat_window_origin])
    center_index = flat_window_origin - zeros_before_origin
    return _map_helper(
        patterns,
        _adp,
        window=boolean_window,
        nav_shape=nav_shape,
        dtype=dtype,
        ignore_map=False,
        center_index=center_index,
        standardize=standardize,
    )[:, :, None, None]

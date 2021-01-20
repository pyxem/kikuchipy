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

"""Private functions for calculating dot products between EBSD patterns
and their neighbours in a map of a 1D or 2D navigation shape.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import generic_filter

from kikuchipy.filters import Window
from kikuchipy.pattern._pattern import _normalize, _zero_mean


# This private module is tested indirectly via the EBSD methods
# get_average_neighbour_dot_product_map() and
# get_neighbour_dot_product_matrices()


def _map_helper(
    patterns: np.ndarray,
    map_function: Callable,
    window: Window,
    nav_shape: tuple,
    dtype_out: np.dtype = np.float32,
    **kwargs,
) -> np.ndarray:
    """Return output of :func:`scipy.ndimage.generic_filter` after
    wrapping the `map_function` to apply at each element of
    `flat_index_map`.
    
    The generic filter function will be applied at each navigation
    point.
    
    Parameters
    ----------
    patterns
        Pattern chunk.
    map_function
        :func:`_neighbour_dot_products`.
    window
        Window defining the pattern neighbours to calculate the dot
        product with.
    nav_shape
        Navigation shape of pattern chunk.
    dtype_out
        Output array data dtype.
    kwargs
        Keyword arguments passed to `map_function`.

    Returns
    -------
    numpy.ndarray
        The generic filter output.
    """
    # Array in the original navigation shape of indices into the
    # flattened navigation axis
    flat_index_map = np.arange(np.prod(nav_shape)).reshape(nav_shape)

    def wrapped_map_function(indices):
        # `indices` contain the indices to be picked out with window
        # from `flat_index_map`
        return map_function(
            patterns=patterns,
            indices=indices.astype(np.int),
            nav_shape=nav_shape,
            dtype_out=dtype_out,
            **kwargs,
        )

    return generic_filter(
        flat_index_map,
        wrapped_map_function,
        footprint=window,
        mode="constant",
        cval=-1,
        output=None if "output" in kwargs else dtype_out,
    )


def _neighbour_dot_products(
    patterns: np.ndarray,
    indices: np.ndarray,
    nav_shape: tuple,
    sig_size: int,
    dtype_out: np.dtype,
    center_index: int,
    zero_mean: bool,
    normalize: bool,
    flat_window_truthy_indices: Optional[np.ndarray] = None,
    output: Optional[np.ndarray] = None,
) -> Union[float, int]:
    """Return either an average of a dot product matrix between a
    pattern and it's neighbours, or the matrix.
    
    Parameters
    ----------
    patterns
        Pattern chunk.
    indices
        Flat array of indices into `patterns`.
    nav_shape
        Navigation shape of `patterns`.
    sig_size
        Size of the signal, i.e. pattern or detector pixels.
    dtype_out
        Data type of the output dot product matrix and also of the
        patterns prior to dot product calculation.
    center_index
        Index into `patterns` for the current pattern to calculate the
        dot products for.
    zero_mean
        Whether to center the pattern intensities by subtracting the 
        mean intensity to get an average intensity of zero,
        individually.
    normalize
        Whether to normalize the pattern intensities to a standard
        deviation of 1 before calculating the dot products. This
        operation is performed after centering the intensities if
        `zero_mean` is True.
    flat_window_truthy_indices
        Flat array of indices into `patterns` for the navigation points
        to calculate dot products with. If None (default), the function
        assumes that both `output` and this parameter is None, and that
        the `output` array of dot products is to be updated inplace.
    output
        A continually, inplace updated 4D array containing the dot
        product matrices.
    """
    # Flat navigation index corresponding to the origin of the window,
    # i.e. into `patterns`, i.e. the current navigation point for which
    # to calculate the dot product with it's neighbours
    pat_idx = indices[center_index]

    # Indices into `indices` of neighbours to compute dot product with,
    # excluding neighbours outside the map and itself
    neighbour_idx = np.where((indices != pat_idx) & (indices != -1))[0]
    neighbours = indices[neighbour_idx]
    neighbours = np.unravel_index(neighbours, nav_shape)
    # Flat array of neighbour patterns
    neighbour_patterns = patterns[neighbours].astype(dtype_out)
    neighbour_patterns = neighbour_patterns.reshape((-1, sig_size))

    # Flat pattern corresponding to the window origin, i.e. the current
    # navigation point to average
    pattern = patterns[np.unravel_index(pat_idx, nav_shape)]
    pattern = pattern.squeeze().flatten().astype(dtype_out)

    # Pre-process pattern intensities
    if zero_mean:
        pattern = _zero_mean(pattern, axis=0)
        neighbour_patterns = _zero_mean(neighbour_patterns, axis=1)
    if normalize:
        pattern = _normalize(pattern, axis=0)
        neighbour_patterns = _normalize(neighbour_patterns, axis=1)

    # Calculate the dot products
    dot_products = neighbour_patterns @ pattern

    if output is None:
        return np.mean(dot_products)
    else:
        center_value = (pattern ** 2).sum()
        output[pat_idx][flat_window_truthy_indices[center_index]] = center_value
        output[pat_idx][
            flat_window_truthy_indices[neighbour_idx]
        ] = dot_products
        # Output variable is modified in place, but `_map_helper()`
        # expects a (in this case discarded) returned value
        return 1


def _get_neighbour_dot_product_matrices(
    patterns: np.ndarray,
    window: Window,
    sig_dim: int,
    sig_size: int,
    zero_mean: bool,
    normalize: bool,
    dtype_out: np.dtype,
) -> np.ndarray:
    """Return  a 4D array of a pattern chunk's navigation shape, and a
    matrix of dot products between a pattern and its neighbours within a
    window in each navigation point in that chunk.

    Parameters
    ----------
    patterns
        Pattern chunk.
    window
        Window defining the neighbours to calculate the average with.
    sig_dim
        Number of signal dimensions.
    sig_size
        Number of pattern pixels.
    zero_mean
        Whether to subtract the mean of each pattern individually to
        center the intensities about zero before calculating the
        dot products.
    normalize
        Whether to normalize the pattern intensities to a standard
        deviation of 1 before calculating the dot products. This
        operation is performed after centering the intensities if
        `zero_mean` is True.
    dtype_out
        Data type of output map.

    Returns
    -------
    adp
        Map of the average dot product between each pattern and its
        neighbours in a chunk of patterns.
    """
    # Get a flat boolean window, a boolean array with True for True
    # window coefficients, and the index of this window's origin
    (
        boolean_window,
        flat_window_truthy_indices,
        center_index,
    ) = _setup_window_indices(window=window)

    nav_shape = patterns.shape[:-sig_dim]
    output = np.empty((np.prod(nav_shape), window.size), dtype=dtype_out)
    output[:] = np.nan

    _map_helper(
        patterns,
        _neighbour_dot_products,
        window=boolean_window,
        nav_shape=nav_shape,
        sig_size=sig_size,
        center_index=center_index,
        flat_window_truthy_indices=flat_window_truthy_indices,
        zero_mean=zero_mean,
        normalize=normalize,
        output=output,
    )

    output = output.reshape(nav_shape + window.shape)

    return output


def _get_average_dot_product_map(
    patterns: np.ndarray,
    window: Window,
    sig_dim: int,
    sig_size: int,
    zero_mean: bool,
    normalize: bool,
    dtype_out: np.dtype,
) -> np.ndarray:
    """Return the average dot product map for a chunk of patterns.

    Parameters
    ----------
    patterns
        Pattern chunk.
    window
        Window defining the neighbours to calculate the average with.
    sig_dim
        Number of signal dimensions.
    sig_size
        Number of pattern pixels.
    zero_mean
        Whether to subtract the mean of each pattern individually to
        center the intensities about zero before calculating the
        dot products.
    normalize
        Whether to normalize the pattern intensities to a standard
        deviation of 1 before calculating the dot products. This
        operation is performed after centering the intensities if
        `zero_mean` is True.
    dtype_out
        Data type of output map.

    Returns
    -------
    adp
        Average dot product map for the chunk of patterns.
    """
    # Get a flat boolean window and the index of this window's origin
    boolean_window, _, center_index = _setup_window_indices(window=window)

    adp = _map_helper(
        patterns,
        _neighbour_dot_products,
        window=boolean_window,
        nav_shape=patterns.shape[:-sig_dim],
        sig_size=sig_size,
        dtype_out=dtype_out,
        center_index=center_index,
        zero_mean=zero_mean,
        normalize=normalize,
    )

    return adp


def _setup_window_indices(window: Window) -> Tuple[np.ndarray, np.ndarray, int]:
    # Index of window origin in flattened window
    flat_window_origin = np.ravel_multi_index(window.origin, window.shape)

    # Make window flat with boolean values
    boolean_window = window.copy().astype(bool)
    flat_window = boolean_window.flatten()

    # Index of window origin in boolean array with only True values
    flat_window_truthy_indices = np.nonzero(flat_window)[0]
    center_index = np.where(flat_window_truthy_indices == flat_window_origin)
    center_index = center_index[0][0]

    return boolean_window, flat_window_truthy_indices, center_index

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
from math import copysign

import numpy as np
from scipy.ndimage import generic_filter
from kikuchipy.indexing.similarity_metrics import (
    SimilarityMetric,
    _zero_mean,
    _normalize,
)
import dask.array as da
from kikuchipy.signals import EBSD, LazyEBSD


def _map_generator(
    signal: Union[LazyEBSD, EBSD],
    map_function: Callable,
    footprint=None,
    **kwargs,
):
    if footprint is None:
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    shape = signal.axes_manager.navigation_shape[::-1]

    flat_index_map = np.arange(np.prod(shape)).reshape(shape)

    def wrapped_map_function(indices):
        # indices picked out with footprint from flat_index_map
        indices = indices.astype(np.int)
        return map_function(signal, indices, shape, **kwargs)

    map = generic_filter(
        flat_index_map,
        wrapped_map_function,
        footprint=footprint,
        mode="constant",
        cval=-1,
        output=np.float64,
    )
    return map


def _adp(
    signal: Union[EBSD, LazyEBSD],
    indices,
    shape,
    center_index,
    subtract_mean,
    dtype,
):
    px = indices[center_index]
    neighbours = indices[np.where((indices != -1) & (indices != px))]
    sig_size = signal.axes_manager.signal_size
    neighbours = np.unravel_index(neighbours, shape)
    signal_data = signal.data if not signal._lazy else signal.data.compute()

    n_data = signal_data[neighbours].reshape((-1, sig_size)).astype(dtype)
    p = (
        signal_data[np.unravel_index(px, shape)]
        .squeeze()
        .flatten()
        .astype(np.float32)
    )
    if subtract_mean:
        p, n_data = _zero_mean(p, n_data, flat=True)
    p, n_data = _normalize(p, n_data, flat=True)
    return np.mean(n_data @ p.T)


def average_dot_product_map(
    signal: Union[EBSD, LazyEBSD],
    footprint: np.ndarray = None,
    center_index: int = 2,
    zero_mean: bool = False,
    dtype=np.float32,
):
    """[summary]

    Parameters
    ----------
    signal : Union[EBSD, LazyEBSD]
        [description]
    footprint : np.ndarray, optional
        [description], by default None
    center_index : int, optional
        [description], by default 2
    zero_mean : bool, optional
        [description], by default False
    dtype : [type], optional
        [description], by default np.float32

    Returns
    -------
    [type]
        [description]
    """
    return _map_generator(
        signal,
        _adp,
        footprint=footprint,
        center_index=center_index,
        subtract_mean=zero_mean,
        dtype=dtype,
    )

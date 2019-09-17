# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import os
import logging
import numpy as np
import dask.array as da

_logger = logging.getLogger(__name__)


def _get_chunks(signal, mbytes_chunk=100):
    """Return suggested data chunks for patterns. Signal axes are not
    chunked. Goals in prioritised order are (i) split into at least as
    many chunks as available CPUs, (ii) limit chunks to approximately
    input mega bytes, `mbytes_chunk`, and (iii) chunk only one
    navigation axis.

    Parameters
    ----------
    signal : kp.signals.EBSD or kp.lazy_signals.LazyEBSD
        Signal with data to chunk.
    mbytes_chunk : int, optional
        Size of chunks in MB, default is 100 MB as suggested in the
        Dask documentation.

    Returns
    -------
    chunks : list
        Suggested chunk size.
    """

    suggested_size = mbytes_chunk * 2 ** 20
    sig_chunks = signal.axes_manager.signal_shape[::-1]
    nav_chunks = np.array(signal.axes_manager.navigation_shape[::-1])
    data_size = signal.data.nbytes
    pattern_size = data_size / nav_chunks.prod()
    num_chunks = np.ceil(data_size / suggested_size)
    i_min, i_max = np.argmin(nav_chunks), np.argmax(nav_chunks)
    cpus = os.cpu_count()
    if num_chunks <= cpus:  # Return approx. as many chunks as CPUs
        nav_chunks[i_max] = nav_chunks[i_max] // cpus
    elif (nav_chunks[i_min] * pattern_size) < suggested_size:
        # Chunk longest navigation axis
        while (nav_chunks.prod() * pattern_size) >= suggested_size:
            nav_chunks[i_max] = np.floor(nav_chunks[i_max] / 1.1)
    else:  # Chunk both navigation axes
        while (nav_chunks.prod() * pattern_size) >= suggested_size:
            i_max = np.argmax(nav_chunks)
            nav_chunks[i_max] = np.floor(nav_chunks[i_max] / 1.1)
    chunks = list(nav_chunks) + list(sig_chunks)
    _logger.info("Suggested chunk size {}".format(chunks))
    return chunks


def _get_dask_array(signal, dtype=None):
    """Return dask array of patterns with appropriate chunking.

    Parameters
    ----------
    signal : kp.signals.EBSD or kp.signals.LazyEBSD
        Signal with data to return dask array from.
    dtype : np.dtype_out, optional
        Data type of returned dask array.

    Returns
    -------
    dask_array : da.array
        Dask array with signal data with appropriate chunking and data
        type.
    """

    if dtype is None:
        dtype = signal.data.dtype
    if signal._lazy or isinstance(signal.data, da.Array):
        dask_array = signal.data
    else:
        sig_chunks = list(signal.axes_manager.signal_shape)[::-1]
        chunks = [8] * len(signal.axes_manager.navigation_shape)
        chunks.extend(sig_chunks)
        dask_array = da.from_array(signal.data, chunks=chunks)
    return dask_array.astype(dtype)

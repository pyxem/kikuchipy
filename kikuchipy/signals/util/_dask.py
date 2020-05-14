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

import dask.array as da
import numpy as np


# This function is not used, but might be useful!
def _get_chunks(data_shape, dtype, mbytes_chunk=100):
    """Return suggested data chunks for patterns.

    Signal axes are not chunked. Goals in prioritised order are (i)
    limit chunks to approximately input mega bytes in ``mbytes_chunk``,
    and (ii) chunk only one navigation axis.

    Parameters
    ----------
    data_shape : tuple of ints
        Shape of data to chunk.
    dtype : :class:`numpy.dtype`
        Data type.
    mbytes_chunk : int, optional
        Size of chunks in MB, default is 100 MB as suggested in the
        Dask documentation.

    Returns
    -------
    chunks : list
        Suggested chunk size.
    """
    if isinstance(data_shape, tuple):
        data_shape = np.array(data_shape)

    suggested_size = mbytes_chunk * 2 ** 20
    sig_chunks = data_shape[-2:]
    nav_chunks = data_shape[:-2]
    data_nbytes = data_shape.prod() * dtype.itemsize
    pattern_size = data_nbytes / nav_chunks.prod()
    i_min, i_max = np.argmin(nav_chunks), np.argmax(nav_chunks)
    if (nav_chunks[i_min] * pattern_size) < suggested_size:
        # Chunk longest navigation axis
        while (nav_chunks.prod() * pattern_size) >= suggested_size:
            nav_chunks[i_max] = np.floor(nav_chunks[i_max] / 1.1)
    else:  # Chunk both navigation axes
        while (nav_chunks.prod() * pattern_size) >= suggested_size:
            i_max = np.argmax(nav_chunks)
            nav_chunks[i_max] = np.floor(nav_chunks[i_max] / 1.1)
    chunks = list(nav_chunks) + list(sig_chunks)

    return chunks


def _get_dask_array(signal, dtype=None):
    """Return dask array of patterns with appropriate chunking.

    Parameters
    ----------
    signal : :class:`~kikuchipy.signals.ebsd.EBSD` or\
            :class:`~kikuchipy.signals.ebsd.LazyEBSD`
        Signal with data to return dask array from.
    dtype : :class:`numpy.dtype`, optional
        Data type of returned dask array.

    Returns
    -------
    dask_array : :class:`dask.array.Array`
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


def _rechunk_learning_results(factors, loadings, mbytes_chunk=100):
    """Return suggested data chunks for learning results.

    It is assumed that the loadings are not transposed. The last axes of
    factors and loadings are not chunked. The aims in prioritised order:
    1. Limit chunks to approximately input MB (``mbytes_chunk``). 2.
    Keep first axis of factors (detector pixels).

    Parameters
    ----------
    factors : :attr:`hyperspy.learn.mva.LearningResults.factors`
        Component patterns in learning results.
    loadings : :attr:`hyperspy.learn.mva.LearningResults.loadings`
        Component loadings in learning results.
    mbytes_chunk : int, optional
        Size of chunks in MB, default is 100 MB as suggested in the Dask
        documentation.

    Returns
    -------
    List of two tuples :
        The first/second tuple are suggested chunks to pass to
        :func:`dask.array.rechunk` for factors/loadings,
        respectively.
    """
    # Make sure the last factors/loading axes have the same shapes
    if factors.shape[-1] != loadings.shape[-1]:
        raise ValueError(
            "The last dimensions in factors and loadings are not the same."
        )

    # Get shape of learning results
    learning_results_shape = factors.shape + loadings.shape

    # Determine maximum number of (strictly necessary) chunks
    suggested_size = mbytes_chunk * 2 ** 20
    factors_size = factors.nbytes
    loadings_size = loadings.nbytes
    total_size = factors_size + loadings_size
    num_chunks = np.ceil(total_size / suggested_size)

    # Get chunk sizes
    if factors_size <= suggested_size:  # Chunk first axis in loadings
        chunks = [(-1, -1), (int(learning_results_shape[2] / num_chunks), -1)]
    else:  # Chunk both first axes
        sizes = [factors_size, loadings_size]
        while (sizes[0] + sizes[1]) >= suggested_size:
            max_idx = int(np.argmax(sizes))
            sizes[max_idx] = np.floor(sizes[max_idx] / 2)
        factors_chunks = int(np.ceil(factors_size / sizes[0]))
        loadings_chunks = int(np.ceil(loadings_size / sizes[1]))
        chunks = [
            (int(learning_results_shape[0] / factors_chunks), -1),
            (int(learning_results_shape[2] / loadings_chunks), -1),
        ]

    return chunks


def _update_learning_results(learning_results, components, dtype_out):
    """Update learning results before calling
    :meth:`hyperspy.learn.mva.MVA.get_decomposition_model` by
    changing data type, keeping only desired components and rechunking
    them into suitable chunks if they are lazy.

    Parameters
    ----------
    learning_results : hyperspy.learn.mva.LearningResults
        Learning results with component patterns and loadings.
    components : None, int or list of ints
        If ``None``, rebuilds the signal from all ``components``. If
        ``int``, rebuilds signal from ``components`` in range 0-given
        ``int``. If list of ``int``, rebuilds signal from only
        ``components`` in given list.
    dtype_out : numpy.float16, numpy.float32 or numpy.float64
        Data type to cast learning results to.

    Returns
    -------
    factors : :attr:`hyperspy.learn.mva.LearningResults.factors`
        Updated component patterns in learning results.
    loadings : :attr:`hyperspy.learn.mva.LearningResults.loadings`
        Updated component loadings in learning results.
    """
    # Change data type
    factors = learning_results.factors.astype(dtype_out)
    loadings = learning_results.loadings.astype(dtype_out)

    # Keep desired components
    if hasattr(components, "__iter__"):  # components is a list of ints
        factors = factors[:, components]
        loadings = loadings[:, components]
    else:  # components is an int
        factors = factors[:, :components]
        loadings = loadings[:, :components]

    # Rechunk if learning results are lazy
    if isinstance(factors, da.Array) and isinstance(loadings, da.Array):
        chunks = _rechunk_learning_results(factors=factors, loadings=loadings)
        factors = factors.rechunk(chunks=chunks[0])
        loadings = loadings.rechunk(chunks=chunks[1])

    return factors, loadings

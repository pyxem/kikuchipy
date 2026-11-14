#
# Copyright 2019-2026 the kikuchipy developers
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
#

import logging
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from kikuchipy.signals.ebsd import EBSD, LazyEBSD

_logger = logging.getLogger(__name__)


def get_chunking(
    signal: "EBSD | LazyEBSD | None" = None,
    data_shape: tuple[int, ...] | None = None,
    nav_dim: int | None = None,
    sig_dim: int | None = None,
    chunk_shape: int | None = None,
    chunk_bytes: int | float | str | None = 30e6,
    dtype: str | np.dtype | type | None = None,
) -> tuple:
    """Get a chunk tuple based on the shape of the signal data.

    The signal dimensions will not be chunked, and the navigation
    dimensions will be chunked based on either ``chunk_shape``, or be
    optimized based on the ``chunk_bytes``.

    This function is inspired by a similar function in :mod:`pyxem`.

    Parameters
    ----------
    signal
        If not given, the following must be: data shape to be chunked,
        ``data_shape``, the number of navigation dimensions,
        ``nav_dim``, the number of signal dimensions, ``sig_dim``, and
        the data array data type ``dtype``.
    data_shape
        Data shape, must be given if ``signal`` is not.
    nav_dim
        Number of navigation dimensions, must be given if ``signal`` is
        not.
    sig_dim
        Number of signal dimensions, must be given if ``signal`` is not.
    chunk_shape
        Shape of navigation chunks. If not given, this size is set
        automatically based on ``chunk_bytes``. This is a rectangle if
        ``signal`` has two navigation dimensions.
    chunk_bytes
        Number of bytes in each chunk. Default is 30e6, i.e. 30 MB.
        Only used if freedom is given to choose, i.e. if ``chunk_shape``
        is not given. Various parameter types are allowed, e.g.
        ``30000000``, ``"30 MB"``, ``"30MiB"``, or the default ``30e6``,
        all resulting in approximately 30 MB chunks.
    dtype
        Data type of the array to chunk. Will take precedence over the
        signal data type if ``signal`` is given. Must be given if
        ``signal`` is not.

    Returns
    -------
    chunks
        Chunk tuple.
    """
    if signal is not None:
        data_shape = signal.data.shape
        nav_dim = signal.axes_manager.navigation_dimension
        sig_dim = signal.axes_manager.signal_dimension
    if dtype is None:
        dtype = signal.data.dtype
    else:
        dtype = np.dtype(dtype)

    chunks_dict = {}
    # Set the desired navigation chunk shape
    for i in range(nav_dim):
        if chunk_shape is None:
            chunks_dict[i] = "auto"
        else:
            chunks_dict[i] = chunk_shape
    # Don't chunk the signal shape
    for i in range(nav_dim, nav_dim + sig_dim):
        chunks_dict[i] = -1

    chunks = da.core.normalize_chunks(
        chunks=chunks_dict,
        shape=data_shape,
        limit=chunk_bytes,
        dtype=dtype,
    )

    return chunks


def get_dask_array(
    signal: "EBSD | LazyEBSD", dtype: str | np.dtype | type | None = None, **kwargs
) -> da.Array:
    """Return dask array of patterns with appropriate chunking.

    Parameters
    ----------
    signal
        Signal with data to return dask array from.
    dtype
        Data type of returned dask array. This is also passed on to
        :func:`~kikuchipy.signals.util.get_chunking`.
    **kwargs
        Keyword arguments passed to
        :func:`~kikuchipy.signals.util.get_chunking` to control the
        number of chunks the output data array is split into. Only
        ``chunk_shape``, ``chunk_bytes`` and ``dtype`` are passed on.

    Returns
    -------
    dask_array
        Dask array with signal data with appropriate chunking and data
        type.
    """
    if dtype is None:
        dtype = signal.data.dtype
    else:
        dtype = np.dtype(dtype)
    if signal._lazy or isinstance(signal.data, da.Array):
        dask_array = signal.data
        if kwargs.pop("rechunk", False):
            new_chunks = _reduce_chunks(
                dask_array=dask_array,
                chunk_bytes=kwargs.pop("chunk_bytes", 8e6),
                dtype_out=dtype,
            )
            dask_array = dask_array.rechunk(new_chunks)
            _logger.info(f"Rechunk Dask array: {dask_array}")
    else:
        chunks = get_chunking(
            signal=signal,
            dtype=dtype,
            chunk_shape=kwargs.pop("chunk_shape", None),
            chunk_bytes=kwargs.pop("chunk_bytes", None),
        )
        dask_array = da.from_array(signal.data, chunks=chunks)
    return dask_array.astype(dtype)


def _reduce_chunks(
    dask_array: da.Array,
    chunk_bytes: int | float = 8e6,
    dtype_out: str | np.dtype | type = "float32",
) -> tuple:
    dtype_out = np.dtype(dtype_out)

    chunksize = dask_array.chunksize
    nav_chunksize = chunksize[:-2]
    nav_ndim = len(nav_chunksize)
    chunks_dict = {i: "auto" for i in range(nav_ndim)}
    chunks_dict.update({i: -1 for i in range(nav_ndim, nav_ndim + 2)})

    if nav_ndim == 2:
        idx_min = np.argmin(nav_chunksize)
        if nav_chunksize[idx_min] * np.prod(chunksize[-2:]) * 4 < chunk_bytes:
            chunks_dict[idx_min] = -1
    chunks = da.core.normalize_chunks(
        chunks=chunks_dict,
        shape=dask_array.shape,
        limit=chunk_bytes,
        dtype=dtype_out,
    )

    old_chunks = dask_array.chunks
    new_chunks = ()
    for i in range(len(chunksize)):
        if chunks_dict[i] == -1:
            new_chunks += (old_chunks[i],)
        else:
            new_chunks += (chunks[i],)

    return new_chunks


def _get_chunk_overlap_depth(window, axes_manager, chunksize: tuple) -> dict:
    """Return overlap depth between navigation chunks equal to the max.
    number of nearest neighbours in each navigation axis.

    Parameters
    ----------
    window : kikuchipy.filters.window.Window
    axes_manager : hyperspy.axes.AxesManager
    chunksize

    Returns
    -------
    overlap_depth
    """
    sig_dim = axes_manager.signal_dimension
    nav_shape = axes_manager.navigation_shape[::-1]
    is_chunked = ~np.equal(chunksize[:-sig_dim], nav_shape)
    overlap_depth = {i: n for i, n in enumerate(window.n_neighbours) if is_chunked[i]}
    return overlap_depth

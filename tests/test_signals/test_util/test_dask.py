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

import dask.array as da
import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.signals.util._dask import get_chunking, get_dask_array


class TestDask:
    def test_get_chunking_no_parameters(self):
        s = kp.signals.LazyEBSD(da.zeros((32, 32, 256, 256), dtype=np.uint16))
        chunks = get_chunking(s)
        assert len(chunks) == 4

    def test_chunk_shape(self):
        s = kp.signals.LazyEBSD(da.zeros((32, 32, 256, 256), dtype=np.uint16))
        assert get_chunking(s, chunk_shape=16) == da.core.normalize_chunks(
            chunks={0: 16, 1: 16, 2: -1, 3: -1},
            limit=30e6,
            shape=s.data.shape,
            dtype=s.data.dtype,
        )

    def test_chunk_bytes(self):
        s = kp.signals.LazyEBSD(da.zeros((32, 32, 256, 256), dtype=np.uint16))
        assert get_chunking(s, chunk_bytes=15e6) == da.core.normalize_chunks(
            chunks={0: "auto", 1: "auto", 2: -1, 3: -1},
            limit=15e6,
            shape=s.data.shape,
            dtype=s.data.dtype,
        )

    def test_get_chunking_dtype(self):
        s = kp.signals.LazyEBSD(da.zeros((32, 32, 256, 256), dtype=np.uint8))
        assert get_chunking(s) == da.core.normalize_chunks(
            chunks={0: "auto", 1: "auto", 2: -1, 3: -1},
            limit=30e6,
            shape=s.data.shape,
            dtype=s.data.dtype,
        )
        assert get_chunking(s, dtype=np.float32) == da.core.normalize_chunks(
            chunks={0: "auto", 1: "auto", 2: -1, 3: -1},
            limit=30e6,
            shape=s.data.shape,
            dtype=np.dtype("float32"),
        )

    @pytest.mark.parametrize(
        "shape, nav_dim, sig_dim, dtype",
        [
            (
                (32, 32, 256, 256),
                2,
                2,
                np.dtype("uint16"),
            ),
            ((32, 32, 256, 256), 2, 2, np.dtype("uint8")),
        ],
    )
    def test_get_chunking_no_signal(self, shape, nav_dim, sig_dim, dtype):
        chunks = get_chunking(
            data_shape=shape, nav_dim=nav_dim, sig_dim=sig_dim, dtype=dtype
        )
        assert chunks == da.core.normalize_chunks(
            chunks={0: "auto", 1: "auto", 2: -1, 3: -1},
            limit=30e6,
            shape=shape,
            dtype=dtype,
        )

    def test_get_dask_array(self):
        s = kp.signals.EBSD((255 * np.random.rand(10, 10, 120, 120)).astype(np.uint8))
        dask_array = get_dask_array(s, chunk_shape=8)
        assert dask_array.chunks == da.core.normalize_chunks(
            chunks={0: 8, 1: 8, 2: -1, 3: -1},
            limit=30e6,
            shape=s.data.shape,
            dtype=s.data.dtype,
        )

        # Make data lazy (chunk size is kept)
        s.data = dask_array.rechunk((5, 5, 120, 120))
        dask_array = get_dask_array(s)
        assert dask_array.chunksize == (5, 5, 120, 120)

    def test_chunk_bytes_indirectly(self):
        s = kp.signals.EBSD(np.zeros((10, 10, 8, 8)))
        array_out0 = get_dask_array(s)
        array_out1 = get_dask_array(s, chunk_bytes="25KiB")
        array_out2 = get_dask_array(s, chunk_bytes=30e3)
        assert array_out0.chunks != array_out1.chunks
        assert array_out1.chunks == array_out2.chunks

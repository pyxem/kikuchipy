# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
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
import pytest

from kikuchipy.signals.util._dask import (
    get_dask_array,
    get_chunking,
    _rechunk_learning_results,
)
from kikuchipy.signals.ebsd import EBSD, LazyEBSD


class TestDask:
    def test_get_chunking_no_parameters(self):
        s = LazyEBSD(da.zeros((32, 32, 256, 256), dtype=np.uint16))
        chunks = get_chunking(s)
        assert len(chunks) == 4

    def test_chunk_shape(self):
        s = LazyEBSD(da.zeros((32, 32, 256, 256), dtype=np.uint16))
        chunks = get_chunking(s, chunk_shape=16)
        assert chunks == ((16, 16), (16, 16), (256,), (256,))

    def test_chunk_bytes(self):
        s = LazyEBSD(da.zeros((32, 32, 256, 256), dtype=np.uint16))
        chunks = get_chunking(s, chunk_bytes=15e6)
        assert chunks == ((8, 8, 8, 8), (8, 8, 8, 8), (256,), (256,))

    def test_get_chunking_dtype(self):
        s = LazyEBSD(da.zeros((32, 32, 256, 256), dtype=np.uint8))
        chunks0 = get_chunking(s, dtype=np.float32)
        chunks1 = get_chunking(s)
        assert chunks0 == ((8, 8, 8, 8), (8, 8, 8, 8), (256,), (256,))
        assert chunks1 == ((16, 16), (16, 16), (256,), (256,))

    @pytest.mark.parametrize(
        "shape, nav_dim, sig_dim, dtype, desired_chunks",
        [
            (
                (32, 32, 256, 256),
                2,
                2,
                np.uint16,
                ((8, 8, 8, 8), (8, 8, 8, 8), (256,), (256,)),
            ),
            (
                (32, 32, 256, 256),
                2,
                2,
                np.uint8,
                ((16, 16), (16, 16), (256,), (256,)),
            ),
        ],
    )
    def test_get_chunking_no_signal(
        self, shape, nav_dim, sig_dim, dtype, desired_chunks
    ):
        chunks = get_chunking(
            data_shape=shape, nav_dim=nav_dim, sig_dim=sig_dim, dtype=dtype,
        )
        assert chunks == desired_chunks

    def test_get_dask_array(self):
        s = EBSD((255 * np.random.rand(10, 10, 120, 120)).astype(np.uint8))
        dask_array = get_dask_array(s, chunk_shape=8)
        assert dask_array.chunksize == (8, 8, 120, 120)

        # Make data lazy
        s.data = dask_array.rechunk((5, 5, 120, 120))
        dask_array = get_dask_array(s)
        assert dask_array.chunksize == (5, 5, 120, 120)

    def test_chunk_bytes_indirectly(self):
        s = EBSD(np.zeros((10, 10, 8, 8)))
        array_out0 = get_dask_array(s)
        array_out1 = get_dask_array(s, chunk_bytes="25KiB")
        array_out2 = get_dask_array(s, chunk_bytes=25e3)
        assert array_out0.chunksize != array_out1.chunksize
        assert array_out1.chunksize == array_out2.chunksize

    def test_rechunk_learning_results(self):
        data = da.from_array(np.random.rand(10, 100, 100, 5).astype(np.float32))
        lazy_signal = LazyEBSD(data)

        # Decomposition
        lazy_signal.decomposition(algorithm="PCA", output_dimension=10)
        factors = lazy_signal.learning_results.factors
        loadings = lazy_signal.learning_results.loadings

        # Raise error when last dimension in factors/loadings are not identical
        with pytest.raises(ValueError, match="The last dimensions in factors"):
            _ = _rechunk_learning_results(factors=factors, loadings=loadings.T)

        # Only chunk first axis in loadings
        chunks = _rechunk_learning_results(
            factors=factors, loadings=loadings, mbytes_chunk=0.02
        )
        assert chunks == [(-1, -1), (200, -1)]

        # Chunk first axis in both loadings and factors
        chunks = _rechunk_learning_results(
            factors=factors, loadings=loadings, mbytes_chunk=0.01
        )
        assert chunks == [(125, -1), (62, -1)]
